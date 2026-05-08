"""
test_sam3_ver2.py

이미지 한 장 + text prompt(들)을 넣으면 SAM3로 masking하고 결과를 PNG로 저장.

Usage:
  python test_sam3_ver2.py \
    --image  /home/dgu/minyoung/Ctrl-World/synthetic_traj/task0004_original.png \
    --prompt "green button and and yellow body" \
    --out    /home/dgu/minyoung/droid_data/tracking/rh20t_0/output.png \
    --ckpt   /home/dgu/minyoung/sam3/checkpoints/sam3.pt \
    --device cuda:1
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/dgu/minyoung/sam3")
from sam3.model_builder import build_sam3_video_predictor

# 프롬프트별 overlay 색상 (RGBA)
PALETTE_RGBA = [
    (0,   200, 100, 140),
    (100, 140, 255, 140),
    (255, 180,  50, 140),
    (220,  60, 220, 140),
    ( 50, 220, 220, 140),
    (255,  80,  80, 140),
]


def overlay_mask(base: Image.Image, mask: np.ndarray, color_rgba: tuple) -> Image.Image:
    """bool mask를 base 이미지 위에 반투명 색상으로 합성."""
    layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
    layer[mask] = color_rgba
    return Image.alpha_composite(base.convert("RGBA"), Image.fromarray(layer))


def draw_label(img: Image.Image, text: str, color_rgb: tuple) -> Image.Image:
    """이미지 좌상단에 텍스트 레이블 추가."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([4, 4, len(text) * 9 + 8, 22], fill=(0, 0, 0, 180))
    draw.text((6, 5), text, fill=color_rgb, font=font)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True, help="입력 이미지 경로")
    parser.add_argument("--prompt",  required=True, nargs="+", help="text prompt (여러 개 가능)")
    parser.add_argument("--out",     default=None,  help="저장 경로 (기본: <image>_masked.png)")
    parser.add_argument("--ckpt",    default="/home/dgu/minyoung/sam3/checkpoints/sam3.pt")
    parser.add_argument("--device",  default="cuda:1")
    args = parser.parse_args()

    if args.out is None:
        base, _ = os.path.splitext(args.image)
        args.out = base + "_masked.png"

    # ── 이미지 로드 ──────────────────────────────────────────────
    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    print(f"이미지: {args.image}  ({W}x{H})")

    # ── SAM3 video predictor (단일 프레임 세션) ──────────────────
    if args.device.startswith("cuda:"):
        gpu_id = int(args.device.split(":")[1])
        gpus = [gpu_id]
    else:
        gpus = None

    print(f"SAM3 로드 중: {args.ckpt}")
    vp = build_sam3_video_predictor(checkpoint_path=args.ckpt, gpus_to_use=gpus)

    tmpdir = tempfile.mkdtemp()
    img.save(os.path.join(tmpdir, "00000.jpg"))

    session_id = vp.start_session(tmpdir)["session_id"]

    # ── 프롬프트별 mask 추출 및 overlay ──────────────────────────
    result = img.convert("RGBA")
    per_prompt_imgs = []

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i, prompt in enumerate(args.prompt):
            color_rgba = PALETTE_RGBA[i % len(PALETTE_RGBA)]
            color_rgb  = color_rgba[:3]

            r    = vp.add_prompt(session_id, frame_idx=0, text=prompt)
            out  = r["outputs"]
            masks = out.get("out_binary_masks", [])
            boxes = out.get("out_boxes_xywh")

            found = len(masks) > 0 and np.array(masks[-1]).any()
            if found:
                mask = np.array(masks[-1]).astype(bool)
                area = int(mask.sum())
                print(f"  [{i}] '{prompt}': 탐지 ✓  area={area}")
            else:
                mask = np.zeros((H, W), dtype=bool)
                print(f"  [{i}] '{prompt}': 탐지 실패")

            # combined overlay
            result = overlay_mask(result, mask, color_rgba)

            # 개별 저장용 이미지
            single = overlay_mask(img.convert("RGBA"), mask, color_rgba)
            single = draw_label(single, prompt, color_rgb)
            per_prompt_imgs.append(single.convert("RGB"))

    vp.close_session(session_id)

    # ── 저장 ────────────────────────────────────────────────────
    result_rgb = result.convert("RGB")
    result_rgb.save(args.out)
    print(f"\n[저장] combined → {args.out}")

    base, ext = os.path.splitext(args.out)
    for i, (prompt, pimg) in enumerate(zip(args.prompt, per_prompt_imgs)):
        fname = f"{base}_{prompt.replace(' ', '_')}{ext}"
        pimg.save(fname)
        print(f"[저장] '{prompt}' → {fname}")


if __name__ == "__main__":
    main()
