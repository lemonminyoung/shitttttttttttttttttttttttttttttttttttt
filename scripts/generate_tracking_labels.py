"""
SAM3 기반 object tracking annotation 생성 스크립트.

result_select_ver2 안의 video.mp4 (exterior_image_2 view)를 읽어
SAM3로 tracking하고, tracking.json + latent.pt + overlay.mp4를 저장한다.

Usage:
  python scripts/generate_tracking_labels.py \
  --clip_model_path checkpoints/clip \
  --svd_model_path  checkpoints/svd


출력 구조:
  TRACKING_ROOT/
    ep0000_Put_the_marker_in_the_pot/
      tracking.json
      latent.pt
      overlay.mp4

수정 방법:
  아래 EPISODE_OBJECTS 딕셔너리에서
  에피소드 폴더명 → object label 리스트를 직접 편집한다.

  "ep0004_Move_the_sharpie_to_the_table": (15, ["sharpie"]),  # 15번 프레임부터
  "ep0000_Put_the_marker_in_the_pot":     (0,  ["marker", "pot"]),  # 처음부터

"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import imageio

# ─────────────────────────────────────────────────────────────────────────────
# ★ 여기를 직접 편집하세요 ★
# 키  : 에피소드 폴더명 (정확히 일치해야 함)
# 값  : (start_frame, [labels])          ← view 생략 시 기본값 1 사용
#        (start_frame, [labels], view)   ← view 명시 (0, 1, 2 중 하나)
#        start_frame — SAM3 initialize를 적용할 프레임 번호 (기본 0)
#        labels      — 추적할 object label 리스트 ("robot arm" 자동 포함)
#        view        — 사용할 카메라 뷰 번호 (기본 1)
# ─────────────────────────────────────────────────────────────────────────────
EPISODE_OBJECTS: dict[str, tuple] = {
    #"0001_pick_view0": (0,  ["blue cube","plate"]),
    #"1799" :(23, ["pen", "green cup"],1),
    #"18599" :(24, ["red bowl", "green plate"], 0),
    "episode_000028": (0,  ["white bowl"]),
    #"episode_000032": (0,  ["duster","orange pen","yellow pencil"]),
    #"episode_000044": (0,  ["green cloth"]),
    #"episode_000041": (0,  ["red bottle"]),
    #"episode_000070": (0,  ["bowl","plate"]),
    #"episode_000118": (0,  ["marker","mug"]),
    #"episode_000123": (0,  ["yellow marker","red mug"]),
    #"episode_000871": (0,  ["green bowl","red bowl"]),
    #"episode_000077": (0,  ["yellow object","plastic bowl"]),
    #"episode_000085": (0,  ["orange can","bowl"]),
    #"episode_000868": (0,  ["cube"]),
    #"episode_000865": (0,  ["pink plate"]),
    #"episode_000855": (0,  ["red block","yellow block"])
}
# ─────────────────────────────────────────────────────────────────────────────
# ep_name → annotation JSON 경로 (선택)
# annotation JSON의 'states' 필드를 action으로, 'texts'를 language_instruction으로 사용
# 없으면 action = [0,0,0,0,0,0,0] zeros로 채움
# ─────────────────────────────────────────────────────────────────────────────
EPISODE_ANNOTATIONS: dict[str, str] = {
    # "ep_name": "/path/to/annotation.json",
    # 예시:
    #"1799": "dataset_example/droid_subset/annotation/val/1799.json",
    #"18599": "dataset_example/droid_subset/annotation/val/18599.json",
    #"0000" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0000.json",
    #"0002" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0002.json",
    #"0003" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0003.json",
    #"0004" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0004.json",
    #"0005" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0005.json",
    #"0006" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0006.json",
    #"episode_000028" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0007.json",
    #"episode_000032" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0007.json",
    #"episode_000044" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0007.json",
    #"episode_000118" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0007.json",
    #"episode_000123" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0007.json",
    #"episode_000871" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0007.json",
    #"episode_000868" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0007.json",
    #"episode_000865" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0007.json",
   #"episode_000855" : "dataset_example/droid_new_setup_full/pickplace/annotation/val/0007.json",


}
# ─────────────────────────────────────────────────────────────────────────────

# 경로 설정
#FRAME_ROOT    = "/home/dgu/minyoung/droid_data/result_select_ver2"   # MP4 소스
#FRAME_ROOT    = "/home/dgu/minyoung/Ctrl-World/dataset_example/droid_new_setup/videos/val/0001" #test
#FRAME_ROOT    = "/home/dgu/minyoung/Ctrl-World/dataset_example/droid_subset/videos/val" #1799 , 18599
#FRAME_ROOT    = "/home/dgu/minyoung/Ctrl-World/dataset_example/droid_new_setup_full/pickplace/videos/val"
FRAME_ROOT    = "/home/dgu/minyoung/datasets/droid_20chunks/videos/chunk-000/observation.images.exterior_1_left"
#FRAME_ROOT    = "/home/dgu/minyoung/Ctrl-World/dataset_example/rh20t/videos/val/0004/"   # MP4 소스 (폴더 단위)
TRACKING_ROOT = "/home/dgu/minyoung/droid_data/tracking"              # 출력
SAM3_CKPT     = "/home/dgu/minyoung/sam3/checkpoints/sam3.pt"
SAM3_DEVICE   = "cuda:1"

# point 정보 JSON 경로 (없거나 항목 없으면 text prompt만 사용)
# 포맷: {ep_name: {obj_name: {"point_coords": [[x,y],...], "point_labels": [1,0,...]}}}
POINT_JSON_PATH = ""

os.makedirs(TRACKING_ROOT, exist_ok=True)

# 객체별 overlay 색상 (RGB)
PALETTE = [
    (255,  80,  80),
    ( 80, 200,  80),
    ( 80, 130, 255),
    (255, 200,  50),
    (220,  80, 220),
    ( 80, 220, 220),
    (255, 160,  40),
    (160, 255, 100),
]


def load_frames_from_mp4(mp4_path: str) -> list:
    """MP4 → list of (H, W, 3) uint8 numpy arrays."""
    reader = imageio.get_reader(mp4_path)
    frames = [np.array(f) for f in reader]
    reader.close()
    return frames


def draw_overlay(frame: np.ndarray, masks_dict: dict,
                 object_labels: list, alpha: float = 0.45) -> np.ndarray:
    vis = frame.astype(np.float32).copy()

    for i, lbl in enumerate(object_labels):
        mask = masks_dict.get(lbl)
        if mask is None or not mask.any():
            continue
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        vis[mask] = vis[mask] * (1 - alpha) + color * alpha
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        vis[y1:y1+2, x1:x2] = color
        vis[y2:y2+2, x1:x2] = color
        vis[y1:y2, x1:x1+2] = color
        vis[y1:y2, x2:x2+2] = color

    result = vis.clip(0, 255).astype(np.uint8)

    pil  = Image.fromarray(result)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, lbl in enumerate(object_labels):
        mask = masks_dict.get(lbl)
        if mask is None or not mask.any():
            continue
        ys, xs = np.where(mask)
        tx, ty = int(xs.min()), max(int(ys.min()) - 16, 2)
        color  = PALETTE[i % len(PALETTE)]
        draw.rectangle([tx-1, ty-1, tx + len(lbl)*8 + 4, ty+15], fill=(0, 0, 0, 180))
        draw.text((tx+2, ty), lbl, fill=color, font=font)

    return np.array(pil)


def build_tracking_json(ep_name, frames, object_labels,
                        sam_manager, clip_model, clip_processor,
                        sam_device, clip_device, start_frame=0,
                        points_dict: dict = None,
                        actions: list = None,           # list of 7-dim lists, len = n_frames after start
                        language_instruction: str = ""):
    from models.object_registry import ObjectRegistry, SHAPE_SIZE

    registry = ObjectRegistry()
    for lbl in object_labels:
        registry.register(lbl)

    # start_frame부터 슬라이싱해서 처음부터인 것처럼 처리
    frames = frames[start_frame:]

    sam_manager.initialize(frames[0], object_labels, points_dict=points_dict)

    frame_records  = []
    overlay_frames = []

    for fidx, frame in enumerate(frames):
        if fidx == 0:
            raw = {lbl: {"mask": sam_manager.object_masks.get(lbl),
                         "absent": sam_manager.object_masks.get(lbl) is None,
                         "cause": None}
                   for lbl in object_labels}
        else:
            chunk_results = sam_manager.update_chunk([frame])
            raw = chunk_results[0]

        masks_dict = {lbl: raw.get(lbl, {}).get("mask") for lbl in object_labels}

        frame_obj_records = {}
        for lbl in object_labels:
            mask    = masks_dict.get(lbl)
            present = mask is not None and mask.any()
            absent  = raw.get(lbl, {}).get("absent", not present)
            cause   = raw.get(lbl, {}).get("cause", None)

            if present:
                bbox_norm  = ObjectRegistry.mask_to_bbox(mask, frame.shape[:2])
                shape_lat  = ObjectRegistry.extract_shape_latent(mask)
                appearance = registry.extract_appearance(
                    frame, mask, clip_model, clip_processor, device=clip_device)
                registry.update(lbl, presence=1.0, appearance=appearance,
                                bbox=bbox_norm, state=0.0,
                                frame=frame, mask=mask, shape_latent=shape_lat)
            else:
                registry.mark_absent(lbl)
                bbox_norm  = np.zeros(4, dtype=np.float32)
                shape_lat  = np.zeros(SHAPE_SIZE * SHAPE_SIZE, dtype=np.float32)
                appearance = np.zeros(512, dtype=np.float32)

            assert len(shape_lat) == 256, (
                f"[{lbl} frame={fidx}] shape_latent must be 256 elements, got {len(shape_lat)}")
            obj = registry.get(lbl)
            frame_obj_records[lbl] = {
                "absent":         absent,
                "cause":          cause,
                "presence":       1.0 if present else 0.0,
                "mask_area":      int(mask.sum()) if present else 0,
                "bad_streak":     0,
                "error_score":    0.0,
                "iou":            None,
                "state":          float(obj.state),
                "shape_score":    float(obj.shape_score),
                "shape_rejected": bool(obj.shape_rejected),
                "area_ratio":     float(obj.area_ratio),
                "extent_ratio":   float(obj.extent_ratio),
                "bbox":           bbox_norm.tolist(),
                "appearance":     appearance.tolist(),
                "shape_latent":   shape_lat.tolist(),
                "mask_crop_16":   shape_lat.reshape(16, 16).tolist(),
            }

        action = actions[fidx] if (actions is not None and fidx < len(actions)) else [0.0] * 7
        frame_records.append({"frame_idx": fidx, "action": action, "objects": frame_obj_records})
        overlay_frames.append(draw_overlay(frame, masks_dict, object_labels))

    tracking = {
        "episode_id":           ep_name,
        "object_labels":        object_labels,
        "language_instruction": language_instruction,
        "frames":               frame_records,
    }
    return tracking, overlay_frames


def print_quality_summary(ep_name: str, tracking: dict,
                          latent_tensor: torch.Tensor):
    """tracking.json 저장 직후 에피소드 품질을 stdout에 출력."""
    frame_records  = tracking["frames"]
    object_labels  = tracking["object_labels"]
    n_track        = len(frame_records)

    print(f"\n  ── quality summary: {ep_name} ──")
    print(f"  tracking frames : {n_track}")

    # per-object stats
    all_empty = [True] * n_track
    for lbl in object_labels:
        presences  = [fr["objects"][lbl]["presence"]  for fr in frame_records]
        mask_areas = [fr["objects"][lbl]["mask_area"] for fr in frame_records]
        n_present  = sum(1 for p in presences if p > 0.5)
        n_empty_obj = n_track - n_present
        ratio      = n_present / max(n_track, 1)
        present_areas = [a for a, p in zip(mask_areas, presences) if p > 0.5]
        avg_area   = float(np.mean(present_areas)) if present_areas else 0.0
        for i, p in enumerate(presences):
            if p > 0.5:
                all_empty[i] = False
        print(f"    {lbl:35s}  presence={ratio:.1%}  "
              f"avg_mask_area={avg_area:6.0f}  empty_frames={n_empty_obj}")

    all_empty_count = sum(all_empty)
    print(f"  frames where ALL objects absent : {all_empty_count}")

    # latent frame count vs tracking frame count
    lat_n  = latent_tensor.shape[0] if latent_tensor is not None else 0
    lat_ok = "OK" if lat_n == n_track else "MISMATCH"
    print(f"  latent_frames={lat_n}  tracking_frames={n_track}  [{lat_ok}]")


def main():
    import argparse
    from transformers import CLIPVisionModelWithProjection, CLIPProcessor
    from config import wm_args

    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_path',   type=str,  default=None)
    parser.add_argument('--svd_model_path',    type=str,  default=None)
    cli = parser.parse_args()

    sys.path.insert(0, "/home/dgu/minyoung/sam3")
    from sam3_manager_new import SAM3ManagerNew as SAM3Manager

    args = wm_args()
    if cli.clip_model_path:
        args.clip_model_path = cli.clip_model_path
    if cli.svd_model_path:
        args.svd_model_path = cli.svd_model_path

    clip_device = "cuda:0"

    # ── init_points JSON 로드 ────────────────────────────────
    all_points = {}
    if os.path.isfile(POINT_JSON_PATH):
        with open(POINT_JSON_PATH) as f:
            all_points = json.load(f)
        print(f"[points] {POINT_JSON_PATH} 로드 — {len(all_points)} 에피소드")
    else:
        print(f"[points] {POINT_JSON_PATH} 없음 → text prompt만 사용")
        print(f"  [권장] overfit 데이터 생성 시 POINT_JSON_PATH를 설정하고 point prompt를 사용하면 "
              f"SAM3 초기화 품질이 크게 향상됩니다.")

    # ── SAM3 ────────────────────────────────────────────────
    sam_manager = SAM3Manager(checkpoint_path=SAM3_CKPT, device=SAM3_DEVICE)

    # ── CLIP ─────────────────────────────────────────────────
    clip_model     = CLIPVisionModelWithProjection.from_pretrained(
        args.clip_model_path, local_files_only=True).to(clip_device).eval()
    clip_processor = CLIPProcessor.from_pretrained(
        args.clip_model_path, local_files_only=True)

    # ── VAE (latent 인코딩) ──────────────────────────────────
    from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
    from torchvision import transforms as T
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.svd_model_path, torch_dtype=torch.float16, local_files_only=True)
    vae = pipeline.vae.to(clip_device).eval()
    to_tensor = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    # ── 에피소드별 처리 ──────────────────────────────────────
    for ep_name, ep_val in EPISODE_OBJECTS.items():
        start_frame = ep_val[0]
        raw_labels  = ep_val[1]
        view_idx    = ep_val[2] if len(ep_val) > 2 else 1   # 기본값 1
        #mp4_path = os.path.join(FRAME_ROOT, ep_name, f"{view_idx}.mp4")
        mp4_path = os.path.join(FRAME_ROOT, f"{ep_name}.mp4")
        if not os.path.isfile(mp4_path):
            print(f"[skip]— {ep_name}.mp4 없음 ({mp4_path})")
            continue

        out_dir      = os.path.join(TRACKING_ROOT, ep_name)
        track_path   = os.path.join(out_dir, "tracking.json")
        latent_path  = os.path.join(out_dir, "latent.pt")
        overlay_path = os.path.join(out_dir, "overlay.mp4")

        if os.path.isfile(track_path) and os.path.isfile(latent_path):
            print(f"[skip] {ep_name} — 이미 존재")
            continue

        os.makedirs(out_dir, exist_ok=True)
        all_labels = list(raw_labels)

        print(f"\n{ep_name}")
        print(f"  start  : frame {start_frame}")
        print(f"  labels : {all_labels}")

        # annotation은 사용하지 않음 (object tracking에 불필요; action은 dataloader가 별도 로드)
        actions_for_ep = None
        lang_instr     = ""

        frames = load_frames_from_mp4(mp4_path)
        print(f"  frames : {len(frames)} (tracking: {len(frames) - start_frame})")

        # tracking
        ep_points = all_points.get(ep_name)   # {obj_name: {point_coords, point_labels}} | None
        if ep_points:
            print(f"  points : {list(ep_points.keys())}")
        sam_manager.reset_session()
        tracking, overlay_frames = build_tracking_json(
            ep_name, frames, all_labels,
            sam_manager, clip_model, clip_processor,
            SAM3_DEVICE, clip_device,
            start_frame=start_frame,
            points_dict=ep_points,
            actions=actions_for_ep,
            language_instruction=lang_instr,
        )
        with open(track_path, "w") as f:
            json.dump(tracking, f)
        print(f"  tracking.json saved")

        imageio.mimwrite(overlay_path, overlay_frames, fps=10)
        print(f"  overlay.mp4  saved")

        # latent — start_frame부터 인코딩해서 tracking.json frame alignment 맞춤
        latent_frames = []
        with torch.no_grad():
            for frame in frames[start_frame:]:
                img_t = to_tensor(Image.fromarray(frame)).unsqueeze(0).half().to(clip_device)
                lat   = vae.encode(img_t).latent_dist.sample() * vae.config.scaling_factor
                latent_frames.append(lat.squeeze(0).cpu().float())
        latent_tensor = torch.stack(latent_frames)
        torch.save(latent_tensor, latent_path)
        print(f"  latent.pt    saved: {latent_tensor.shape}")

        print_quality_summary(ep_name, tracking, latent_tensor)

    print(f"\n완료 → {TRACKING_ROOT}")


if __name__ == "__main__":
    main()
