#!/usr/bin/env python3
"""
test_vae_masked_latent_deformation.py

Ctrl-World VAE masked latent distance로 object deformation 감지 테스트.
LPIPS/DINO 없이 VAE latent 비교만 수행한다.

Usage:
  python scripts/test_vae_masked_latent_deformation.py \
    --first_real_frame synthetic_traj/Rollouts_interact_pi/video/init_frame_traj85_view0.png \
    --generated_frame_dir synthetic_traj/Rollouts_interact_pi/video/droid_tracking_..._orange_can.mp4_or_dir \
    --mask_json synthetic_traj/.../..._orange_can.json \
    --label "orange can" \
    --output_dir debug_vae_latent_deform/traj85 \
    --svd_model_path checkpoints/svd \
    --device cuda:0
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── VAE 로드 ─────────────────────────────────────────────────────────────────

def load_vae(svd_model_path: str, device: str, dtype: torch.dtype):
    from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
    print(f"[VAE] loading from {svd_model_path} ...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(svd_model_path)
    vae = pipe.vae.to(device).to(dtype)
    vae.eval()
    del pipe
    print(f"[VAE] loaded  scaling_factor={vae.config.scaling_factor}")
    return vae


# ── 마스크 유틸 ───────────────────────────────────────────────────────────────

def bbox_to_mask(bbox, H, W):
    """[x1,y1,x2,y2] pixel → (H,W) bool mask."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def load_mask_dir(mask_dir, frame_idx, H, W):
    """mask_dir 안의 {frame_idx:05d}.png 또는 {frame_idx}.png 로드."""
    for name in [f"{frame_idx:05d}.png", f"{frame_idx}.png",
                 f"mask_{frame_idx:05d}.png", f"mask_{frame_idx}.png"]:
        p = os.path.join(mask_dir, name)
        if os.path.exists(p):
            m = np.array(Image.open(p).convert("L").resize((W, H), Image.NEAREST))
            return m > 127
    return None


# ── 핵심 함수 ─────────────────────────────────────────────────────────────────

def extract_masked_vae_object_latent(frame_uint8, mask_bool, vae, device, dtype,
                                     pad_ratio=2.0, crop_size=224):
    """
    frame_uint8 : (H, W, 3) np.uint8
    mask_bool   : (H, W) bool  (None이면 전체 사용)
    반환        : (C,) float32 cpu tensor (masked-average pooled latent vec)
    """
    H, W = frame_uint8.shape[:2]

    # ── bbox 계산 및 padded square crop ──────────────────────────────────────
    if mask_bool is not None and mask_bool.any():
        ys, xs = np.where(mask_bool)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    else:
        x1, y1, x2, y2 = 0, 0, W, H

    obj_w, obj_h = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(obj_w, obj_h) * pad_ratio / 2.0

    # boundary clamp
    lx = max(0, int(cx - half))
    rx = min(W, int(cx + half))
    ty = max(0, int(cy - half))
    by = min(H, int(cy + half))

    # 실제 크기 맞춤 (clamp 후 정사각형 보장)
    side = min(rx - lx, by - ty)
    rx, by = lx + side, ty + side
    if rx > W:
        lx, rx = W - side, W
    if by > H:
        ty, by = H - side, H

    frame_crop = frame_uint8[ty:by, lx:rx]
    if mask_bool is not None:
        mask_crop  = mask_bool[ty:by, lx:rx]
    else:
        mask_crop  = np.ones((by - ty, rx - lx), dtype=bool)

    # ── resize to crop_size ───────────────────────────────────────────────────
    frame_resized = cv2.resize(frame_crop, (crop_size, crop_size),
                               interpolation=cv2.INTER_LINEAR)
    mask_resized  = cv2.resize(mask_crop.astype(np.uint8) * 255,
                               (crop_size, crop_size),
                               interpolation=cv2.INTER_NEAREST) > 127

    # ── VAE encode ────────────────────────────────────────────────────────────
    x = torch.from_numpy(frame_resized).to(dtype).to(device)
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0 * 2.0 - 1.0   # (1,3,H,W) [-1,1]

    with torch.no_grad():
        latent = vae.encode(x).latent_dist.sample()             # (1,4,lH,lW)
        latent = latent * vae.config.scaling_factor

    lH, lW = latent.shape[2], latent.shape[3]

    # ── mask → latent 해상도 nearest downsample ───────────────────────────────
    mask_t = torch.from_numpy(mask_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_lat = F.interpolate(mask_t, size=(lH, lW), mode='nearest').squeeze() > 0.5   # (lH, lW)

    # ── masked average pooling ────────────────────────────────────────────────
    lat = latent.squeeze(0)   # (4, lH, lW)
    if mask_lat.any():
        vec = lat[:, mask_lat].mean(dim=1)   # (4,)
    else:
        vec = lat.flatten(1).mean(dim=1)     # fallback: 전체 평균

    return vec.cpu().float(), (lx, ty, rx, by), frame_resized, mask_resized


# ── 거리 계산 ─────────────────────────────────────────────────────────────────

def cosine_dist(a, b):
    a, b = F.normalize(a.unsqueeze(0), dim=1), F.normalize(b.unsqueeze(0), dim=1)
    return float(1.0 - (a * b).sum())


def l2_dist(a, b):
    return float((a - b).norm())


# ── 프레임 로더 ───────────────────────────────────────────────────────────────

def load_frames_from_source(src):
    """mp4 파일 또는 프레임 디렉토리 → list of (H,W,3) uint8."""
    if os.path.isfile(src) and src.endswith('.mp4'):
        cap = cv2.VideoCapture(src)
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f[:, :, ::-1].copy())  # BGR→RGB
        cap.release()
        return frames
    elif os.path.isdir(src):
        exts = ('.png', '.jpg', '.jpeg')
        paths = sorted(p for p in os.listdir(src) if p.lower().endswith(exts))
        return [np.array(Image.open(os.path.join(src, p)).convert('RGB')) for p in paths]
    else:
        raise ValueError(f"generated_frame_dir: {src} 는 mp4 파일이나 디렉토리여야 합니다")


# ── 시각화 ────────────────────────────────────────────────────────────────────

def make_contact_sheet(images, n_cols=10, cell_size=112, title=None):
    """images: list of (H,W,3) uint8 → contact sheet PIL Image."""
    n = len(images)
    n_rows = math.ceil(n / n_cols)
    sheet = np.zeros((n_rows * cell_size + (30 if title else 0),
                      n_cols * cell_size, 3), dtype=np.uint8)
    offset_y = 30 if title else 0
    for i, img in enumerate(images):
        r, c = divmod(i, n_cols)
        thumb = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        sheet[offset_y + r*cell_size: offset_y + (r+1)*cell_size,
              c*cell_size: (c+1)*cell_size] = thumb
    pil = Image.fromarray(sheet)
    if title:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        draw.text((4, 4), title, fill=(255, 255, 200), font=font)
    return pil


def draw_dist_curve(cosine_dists, l2_dists, output_path):
    """latent distance curve PNG 저장."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(cosine_dists, 'b-o', markersize=3, label='cosine dist')
        ax1.set_ylabel('cosine distance')
        ax1.set_title('VAE Masked Latent Distance vs Anchor Frame')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax2.plot(l2_dists, 'r-o', markersize=3, label='L2 dist')
        ax2.set_ylabel('L2 distance')
        ax2.set_xlabel('frame')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[SAVE] {output_path}")
    except ImportError:
        print("[WARN] matplotlib 없음 - distance curve 저장 생략")


# ── 마스크 오버레이 ───────────────────────────────────────────────────────────

def overlay_mask_on_frame(frame, mask, color=(255, 80, 80), alpha=0.45):
    overlay = frame.copy().astype(float)
    if mask is not None and mask.any():
        for c, v in enumerate(color):
            overlay[:, :, c] = np.where(mask,
                                        overlay[:, :, c] * (1 - alpha) + v * alpha,
                                        overlay[:, :, c])
    return overlay.clip(0, 255).astype(np.uint8)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_real_frame',    type=str, default=None)
    parser.add_argument('--real_frame_dir',      type=str, default=None)
    parser.add_argument('--generated_frame_dir', type=str, required=True)
    parser.add_argument('--mask_json',           type=str, default=None,
                        help='tracking JSON (bbox per frame)')
    parser.add_argument('--mask_dir',            type=str, default=None,
                        help='per-frame mask PNG directory')
    parser.add_argument('--label',               type=str, required=True)
    parser.add_argument('--output_dir',          type=str, default='debug_vae_latent_deform')
    parser.add_argument('--svd_model_path',      type=str,
                        default='/home/dgu/minyoung/Ctrl-World/checkpoints/svd')
    parser.add_argument('--device',              type=str, default='cuda:0')
    parser.add_argument('--dtype',               type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--pad_ratio',           type=float, default=2.0)
    parser.add_argument('--crop_size',           type=int, default=224)
    parser.add_argument('--n_cols',              type=int, default=10,
                        help='contact sheet columns')
    args = parser.parse_args()

    dtype_map = {'float32': torch.float32, 'float16': torch.float16,
                 'bfloat16': torch.bfloat16}
    dtype = dtype_map[args.dtype]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── VAE 로드 ──────────────────────────────────────────────────────────────
    vae = load_vae(args.svd_model_path, args.device, dtype)

    # ── 마스크 소스 파싱 ──────────────────────────────────────────────────────
    tracking_frames = None
    if args.mask_json is not None:
        with open(args.mask_json) as f:
            tj = json.load(f)
        tracking_frames = tj.get('frames', [])
        print(f"[MASK] mask_json loaded: {len(tracking_frames)} frames  label={tj.get('label')}")
    elif args.mask_dir is not None:
        print(f"[MASK] mask_dir={args.mask_dir}")
    else:
        print("[MASK] 마스크 없음 — 전체 frame 사용")

    # ── first real frame 로드 ─────────────────────────────────────────────────
    if args.first_real_frame is not None:
        first_frame = np.array(Image.open(args.first_real_frame).convert('RGB'))
    elif args.real_frame_dir is not None:
        exts = ('.png', '.jpg', '.jpeg')
        paths = sorted(p for p in os.listdir(args.real_frame_dir)
                       if p.lower().endswith(exts))
        first_frame = np.array(Image.open(os.path.join(args.real_frame_dir, paths[0])).convert('RGB'))
    else:
        raise ValueError("--first_real_frame 또는 --real_frame_dir 필요")

    H, W = first_frame.shape[:2]
    print(f"[FRAME] first_real_frame: {W}×{H}")

    # ── anchor mask (frame 0) ─────────────────────────────────────────────────
    if tracking_frames is not None:
        anchor_bbox = tracking_frames[0]['bbox']   # [x1,y1,x2,y2]
        anchor_mask = bbox_to_mask(anchor_bbox, H, W)
    elif args.mask_dir is not None:
        anchor_mask = load_mask_dir(args.mask_dir, 0, H, W)
        anchor_bbox = None
    else:
        anchor_mask = None
        anchor_bbox = None

    # ── anchor latent 추출 ────────────────────────────────────────────────────
    anchor_vec, anchor_crop_box, anchor_crop_img, anchor_mask_crop = \
        extract_masked_vae_object_latent(
            first_frame, anchor_mask, vae, args.device, dtype,
            pad_ratio=args.pad_ratio, crop_size=args.crop_size)
    print(f"[ANCHOR] vec shape={anchor_vec.shape}  "
          f"crop_box={anchor_crop_box}  "
          f"mask_area={anchor_mask.sum() if anchor_mask is not None else 'N/A'}")

    # anchor crop 저장
    Image.fromarray(anchor_crop_img).save(os.path.join(args.output_dir, 'anchor_crop.png'))
    ov = overlay_mask_on_frame(first_frame, anchor_mask)
    Image.fromarray(ov).save(os.path.join(args.output_dir, 'anchor_mask_overlay.png'))

    # ── generated frames 로드 ─────────────────────────────────────────────────
    gen_frames = load_frames_from_source(args.generated_frame_dir)
    print(f"[GEN] {len(gen_frames)} frames loaded")

    # ── per-frame 비교 ────────────────────────────────────────────────────────
    results = []
    crop_imgs, frame_imgs, mask_overlay_imgs = [], [], []

    for t, gen_frame in enumerate(gen_frames):
        gH, gW = gen_frame.shape[:2]

        # mask 획득
        if tracking_frames is not None:
            if t < len(tracking_frames):
                tf = tracking_frames[t]
                bbox_t = tf['bbox']
                mask_t = bbox_to_mask(bbox_t, gH, gW)
                area_t = float(tf.get('area', mask_t.sum()))
                bbox_out = bbox_t
                absent_t = tf.get('absent', False)
                cause_t  = tf.get('cause', None)
            else:
                mask_t  = None
                area_t  = 0.0
                bbox_out = None
                absent_t = True
                cause_t  = 'no_tracking_data'
        elif args.mask_dir is not None:
            mask_t  = load_mask_dir(args.mask_dir, t, gH, gW)
            area_t  = float(mask_t.sum()) if mask_t is not None else 0.0
            bbox_out = None
            absent_t = (mask_t is None or not mask_t.any())
            cause_t  = None
        else:
            mask_t  = None
            area_t  = float(gH * gW)
            bbox_out = [0, 0, gW, gH]
            absent_t = False
            cause_t  = None

        cur_vec, crop_box, crop_img, mask_crop = \
            extract_masked_vae_object_latent(
                gen_frame, mask_t, vae, args.device, dtype,
                pad_ratio=args.pad_ratio, crop_size=args.crop_size)

        cos_d = cosine_dist(anchor_vec, cur_vec)
        l2_d  = l2_dist(anchor_vec, cur_vec)
        anchor_area = float(anchor_mask.sum()) if anchor_mask is not None else float(H * W)
        area_ratio  = area_t / (anchor_area + 1e-6)

        # shape_score: tracking JSON에서 읽거나 N/A
        shape_score = None
        if tracking_frames is not None and t < len(tracking_frames):
            shape_score = tracking_frames[t].get('shape_score')

        rec = {
            'frame': t,
            'latent_cosine_dist': round(cos_d, 6),
            'latent_l2_dist':     round(l2_d, 6),
            'area':               round(area_t, 1),
            'area_ratio':         round(area_ratio, 4),
            'shape_score':        shape_score,
            'bbox':               bbox_out,
            'absent':             absent_t,
            'cause':              cause_t,
        }
        results.append(rec)
        print(f"  frame {t:3d}: cos={cos_d:.4f}  l2={l2_d:.4f}  "
              f"area_ratio={area_ratio:.3f}  absent={absent_t}  cause={cause_t}")

        # 시각화 수집
        crop_imgs.append(crop_img)
        frame_imgs.append(cv2.resize(gen_frame, (W, H), interpolation=cv2.INTER_AREA)
                          if (gH != H or gW != W) else gen_frame)
        ov_t = overlay_mask_on_frame(
            cv2.resize(gen_frame, (W, H), interpolation=cv2.INTER_AREA) if (gH != H or gW != W) else gen_frame,
            cv2.resize(mask_t.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if mask_t is not None else None
        )
        mask_overlay_imgs.append(ov_t)

    # ── metrics.json 저장 ─────────────────────────────────────────────────────
    metrics = {
        'label':        args.label,
        'anchor_frame': args.first_real_frame or args.real_frame_dir,
        'anchor_mask_area': float(anchor_mask.sum()) if anchor_mask is not None else None,
        'anchor_bbox':  anchor_crop_box,
        'pad_ratio':    args.pad_ratio,
        'crop_size':    args.crop_size,
        'total_frames': len(results),
        'frames':       results,
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] {metrics_path}")

    # ── distance curve ─────────────────────────────────────────────────────────
    cos_dists = [r['latent_cosine_dist'] for r in results]
    l2_dists  = [r['latent_l2_dist']     for r in results]
    draw_dist_curve(cos_dists, l2_dists,
                    os.path.join(args.output_dir, 'latent_distance_curve.png'))

    # ── contact sheets ─────────────────────────────────────────────────────────
    cs_crop = make_contact_sheet(crop_imgs,         n_cols=args.n_cols,
                                 cell_size=112, title=f"crop ({args.label})")
    cs_frame = make_contact_sheet(frame_imgs,        n_cols=args.n_cols,
                                  cell_size=112, title=f"frame ({args.label})")
    cs_mask  = make_contact_sheet(mask_overlay_imgs, n_cols=args.n_cols,
                                  cell_size=112, title=f"mask overlay ({args.label})")

    cs_crop.save(os.path.join(args.output_dir, 'crop_contact_sheet.png'))
    cs_frame.save(os.path.join(args.output_dir, 'frame_contact_sheet.png'))
    cs_mask.save(os.path.join(args.output_dir,  'mask_overlay_contact_sheet.png'))
    print(f"[SAVE] contact sheets → {args.output_dir}/")

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  total frames : {len(results)}")
    print(f"  cosine dist  : min={min(cos_dists):.4f}  max={max(cos_dists):.4f}  "
          f"mean={sum(cos_dists)/len(cos_dists):.4f}")
    print(f"  l2 dist      : min={min(l2_dists):.4f}   max={max(l2_dists):.4f}   "
          f"mean={sum(l2_dists)/len(l2_dists):.4f}")
    print(f"  output_dir   : {args.output_dir}")
    print("────────────────────────────────────────────────────────────────────")


if __name__ == '__main__':
    main()
