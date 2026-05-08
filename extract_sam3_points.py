"""
extract_sam3_points.py

annotated image에서 색깔 point marker를 검출하여
SAM3 Tracker 입력 형식(input_points / input_labels)으로 변환 저장.

Usage:
  python extract_sam3_points.py \
    --original_image  /home/dgu/minyoung/droid_data/result_select_ver2/ep0000_Put_the_marker_in_the_pot/frame_0042.png \
    --annotated_image /home/dgu/minyoung/droid_data/select_point_dataset/Task_0_frame_0042.png \
    --out_json        /home/dgu/minyoung/droid_data/output.json \
    --out_npy         /home/dgu/minyoung/droid_data/output \
    --debug_dir       /home/dgu/minyoung/droid_data/debug/ \
    --color_scheme    red_blue \
    --min_area        10 \
    --max_area        1000
"""

import argparse
import json
import os
import sys
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. I/O helpers
# ─────────────────────────────────────────────────────────────

def read_image(path: str) -> np.ndarray:
    """BGR uint8 이미지 로드. 없으면 FileNotFoundError."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다 (손상 또는 포맷 불일치): {path}")
    return img


# ─────────────────────────────────────────────────────────────
# 2. Color masking
# ─────────────────────────────────────────────────────────────

def build_color_masks(
    hsv: np.ndarray,
    color_scheme: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    HSV 이미지에서 positive/negative 마스크 반환.
    returns: (positive_mask, negative_mask) — uint8 binary
    """
    if color_scheme == "magenta_cyan":
        # positive: magenta/pink  H 140~170
        pos = cv2.inRange(hsv,
                          np.array([140, 80, 80]),
                          np.array([170, 255, 255]))
        # negative: cyan          H 80~100
        neg = cv2.inRange(hsv,
                          np.array([80, 80, 80]),
                          np.array([100, 255, 255]))

    elif color_scheme == "red_blue":
        # positive: red (hue 0~10 + 170~180)
        red_lo = cv2.inRange(hsv,
                             np.array([0,   80, 80]),
                             np.array([10,  255, 255]))
        red_hi = cv2.inRange(hsv,
                             np.array([170, 80, 80]),
                             np.array([180, 255, 255]))
        pos = cv2.bitwise_or(red_lo, red_hi)
        # negative: blue          H 100~130
        neg = cv2.inRange(hsv,
                          np.array([100, 80, 80]),
                          np.array([130, 255, 255]))
    else:
        raise ValueError(f"지원하지 않는 color_scheme: {color_scheme}. "
                         f"'magenta_cyan' 또는 'red_blue' 중 선택하세요.")

    return pos, neg


# ─────────────────────────────────────────────────────────────
# 3. Morphology cleanup
# ─────────────────────────────────────────────────────────────

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Open → Close morphology로 노이즈 제거."""
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


# ─────────────────────────────────────────────────────────────
# 4. Centroid extraction
# ─────────────────────────────────────────────────────────────

def extract_centroids(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    label_name: str,
) -> list[list[int]]:
    """
    Connected components에서 area 조건을 만족하는 blob의 centroid 반환.
    returns: [[x, y], ...]
    """
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    points = []
    for i in range(1, n_labels):          # 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            cx = int(round(centroids[i][0]))
            cy = int(round(centroids[i][1]))
            points.append([cx, cy])
            print(f"  [{label_name}] blob #{i}: area={area}  centroid=({cx}, {cy})")
        else:
            print(f"  [{label_name}] blob #{i}: area={area}  → 범위 밖, 건너뜀")

    return points


# ─────────────────────────────────────────────────────────────
# 5. SAM3 format builder
# ─────────────────────────────────────────────────────────────

def build_sam3_format(
    pos_points: list[list[int]],
    neg_points: list[list[int]],
    original_path: str,
    annotated_path: str,
    W: int,
    H: int,
) -> dict:
    """
    point_coords / point_labels 및 SAM3 Tracker용 중첩 format 생성.
    sam3_input_points shape: [image_idx][object_idx][point_idx][xy]
    """
    point_coords = pos_points + neg_points
    point_labels = [1] * len(pos_points) + [0] * len(neg_points)

    return {
        "original_image":    original_path,
        "annotated_image":   annotated_path,
        "image_size":        {"width": W, "height": H},
        "point_coords":      point_coords,
        "point_labels":      point_labels,
        "sam3_input_points": [[point_coords]],   # [1][1][N][2]
        "sam3_input_labels": [[point_labels]],   # [1][1][N]
        "format_note": (
            "sam3_input_points shape = "
            "[image_index][object_index][point_index][xy]; "
            "labels: 1=positive, 0=negative"
        ),
    }


# ─────────────────────────────────────────────────────────────
# 6. Save outputs
# ─────────────────────────────────────────────────────────────

def save_outputs(
    data: dict,
    out_json: str,
    out_npy: Optional[str],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[저장] JSON → {out_json}")

    if out_npy:
        coords  = np.array(data["point_coords"],      dtype=np.int32)
        labels  = np.array(data["point_labels"],      dtype=np.int32)
        s3pts   = np.array(data["sam3_input_points"], dtype=np.int32)
        s3lbls  = np.array(data["sam3_input_labels"], dtype=np.int32)

        np.save(f"{out_npy}_point_coords.npy",      coords)
        np.save(f"{out_npy}_point_labels.npy",      labels)
        np.save(f"{out_npy}_sam3_input_points.npy", s3pts)
        np.save(f"{out_npy}_sam3_input_labels.npy", s3lbls)
        print(f"[저장] NPY  → {out_npy}_point_coords.npy  외 3개")


# ─────────────────────────────────────────────────────────────
# 7. Debug outputs
# ─────────────────────────────────────────────────────────────

def save_debug_outputs(
    debug_dir: str,
    annotated_bgr: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    pos_points: list[list[int]],
    neg_points: list[list[int]],
    data: dict,
) -> None:
    os.makedirs(debug_dir, exist_ok=True)

    cv2.imwrite(os.path.join(debug_dir, "positive_mask.png"), pos_mask)
    cv2.imwrite(os.path.join(debug_dir, "negative_mask.png"), neg_mask)

    # overlay: annotated 이미지 위에 centroid 표시
    overlay = annotated_bgr.copy()
    for i, (x, y) in enumerate(pos_points):
        cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)         # green
        cv2.putText(overlay, f"P{i}", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for i, (x, y) in enumerate(neg_points):
        cv2.circle(overlay, (x, y), 8, (0, 0, 255), -1)         # red
        cv2.putText(overlay, f"N{i}", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(debug_dir, "detected_points_overlay.png"), overlay)

    summary = {
        "positive_count": len(pos_points),
        "negative_count": len(neg_points),
        "positive_points": pos_points,
        "negative_points": neg_points,
        "image_size": data["image_size"],
    }
    with open(os.path.join(debug_dir, "debug_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[디버그] → {debug_dir}/")
    print(f"         positive_mask.png / negative_mask.png")
    print(f"         detected_points_overlay.png / debug_summary.json")


# ─────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotated image에서 색깔 point를 검출해 SAM3 형식으로 저장")
    parser.add_argument("--original_image",  required=True,
                        help="원본 이미지 경로 (SAM3 입력용)")
    parser.add_argument("--annotated_image", required=True,
                        help="색깔 점이 찍힌 이미지 경로 (좌표 추출용)")
    parser.add_argument("--out_json",        required=True,
                        help="출력 JSON 경로")
    parser.add_argument("--out_npy",         default=None,
                        help="NPY 저장 경로 prefix (선택)")
    parser.add_argument("--debug_dir",       default=None,
                        help="디버그 이미지 저장 폴더 (선택)")
    parser.add_argument("--color_scheme",    default="magenta_cyan",
                        choices=["magenta_cyan", "red_blue"],
                        help="포인트 색상 규칙 (기본: magenta_cyan)")
    parser.add_argument("--min_area",        type=int, default=10,
                        help="blob 최소 면적 (기본: 10)")
    parser.add_argument("--max_area",        type=int, default=1000,
                        help="blob 최대 면적 (기본: 1000)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"original_image  : {args.original_image}")
    print(f"annotated_image : {args.annotated_image}")
    print(f"color_scheme    : {args.color_scheme}")
    print(f"area range      : [{args.min_area}, {args.max_area}]")
    print("=" * 60)

    # ── 이미지 로드 ──────────────────────────────────────────
    orig_bgr  = read_image(args.original_image)
    annot_bgr = read_image(args.annotated_image)

    H_o, W_o = orig_bgr.shape[:2]
    H_a, W_a = annot_bgr.shape[:2]
    if (H_o, W_o) != (H_a, W_a):
        raise ValueError(
            f"이미지 크기 불일치: original={W_o}x{H_o}, "
            f"annotated={W_a}x{H_a}\n"
            "검출된 좌표를 SAM3에 그대로 사용하려면 두 이미지 크기가 같아야 합니다."
        )
    print(f"이미지 크기 확인: {W_o}x{H_o} ✓")

    # ── 색깔 마스크 생성 ─────────────────────────────────────
    diff = cv2.absdiff(annot_bgr, orig_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # annotated image에서 새로 바뀐 픽셀만 후보로 사용
    _, changed_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(annot_bgr, cv2.COLOR_BGR2HSV)
    pos_mask_raw, neg_mask_raw = build_color_masks(hsv, args.color_scheme)

    # 색상 조건 AND 변경된 픽셀 조건
    pos_mask_raw = cv2.bitwise_and(pos_mask_raw, changed_mask)
    neg_mask_raw = cv2.bitwise_and(neg_mask_raw, changed_mask)

    pos_mask = clean_mask(pos_mask_raw)
    neg_mask = clean_mask(neg_mask_raw)

    # ── Centroid 추출 ────────────────────────────────────────
    print("\n[Positive points]")
    pos_points = extract_centroids(pos_mask, args.min_area, args.max_area, "positive")
    print(f"  → {len(pos_points)}개 검출")

    print("\n[Negative points]")
    neg_points = extract_centroids(neg_mask, args.min_area, args.max_area, "negative")
    print(f"  → {len(neg_points)}개 검출")

    total = len(pos_points) + len(neg_points)
    if total == 0:
        raise RuntimeError(
            "point가 하나도 검출되지 않았습니다. "
            "color_scheme / HSV 범위 / min_area 파라미터를 확인하세요."
        )
    if total > 20:
        print(f"\n⚠️  WARNING: point가 {total}개 검출됐습니다. "
              "오검출 가능성이 있으니 debug image를 반드시 확인하세요.")

    print(f"\n⚠️  원본 이미지에 비슷한 색이 포함돼 있으면 오검출될 수 있습니다. "
          "debug image를 반드시 확인하세요.")

    # ── SAM3 포맷 빌드 ───────────────────────────────────────
    data = build_sam3_format(
        pos_points, neg_points,
        args.original_image, args.annotated_image,
        W_o, H_o,
    )

    # ── 저장 ────────────────────────────────────────────────
    save_outputs(data, args.out_json, args.out_npy)

    if args.debug_dir:
        save_debug_outputs(
            args.debug_dir, annot_bgr,
            pos_mask, neg_mask,
            pos_points, neg_points, data,
        )

    print("\n[결과 요약]")
    print(f"  positive points : {len(pos_points)}")
    print(f"  negative points : {len(neg_points)}")
    print(f"  point_coords    : {data['point_coords']}")
    print(f"  point_labels    : {data['point_labels']}")
    print("완료.")


if __name__ == "__main__":
    main()


# =============================================================================
# SAM3 Tracker에 넣는 예시 코드
# =============================================================================
#
# import json, torch
# from PIL import Image
# from transformers import Sam3TrackerModel, Sam3TrackerProcessor
#
# # 1) JSON 로드
# with open("output.json") as f:
#     data = json.load(f)
#
# # 2) 원본 이미지 로드 (annotated_image가 아니라 original_image)
# image = Image.open(data["original_image"]).convert("RGB")
#
# # 3) Processor & Model 로드
# processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
# model     = Sam3TrackerModel.from_pretrained("facebook/sam3").to("cuda")
#
# # 4) 입력 생성
# inputs = processor(
#     images=image,
#     input_points=data["sam3_input_points"],   # [[[x1,y1],[x2,y2],...]]
#     input_labels=data["sam3_input_labels"],   # [[[1, 1, 0, ...]]]
#     return_tensors="pt",
# ).to("cuda")
#
# # 5) 추론
# with torch.no_grad():
#     outputs = model(**inputs, multimask_output=True)
#
# # 6) 마스크 추출
# masks = processor.post_process_masks(
#     outputs.pred_masks,
#     inputs["original_sizes"],
#     inputs["reshaped_input_sizes"],
# )
# =============================================================================
