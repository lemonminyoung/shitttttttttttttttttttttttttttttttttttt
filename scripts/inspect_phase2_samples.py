"""
scripts/inspect_phase2_samples.py
Phase2 데이터셋 sanity inspection.

selected_samples.json에서 무작위 K개를 뽑아 각 sample의
  bad_gen / gt_real / last_good / mask overlay / bbox / action curve
를 contact sheet로 저장한다.

이 inspection을 통과하기 전에 Phase2 학습을 시작하지 마라.

사용법:
    python scripts/inspect_phase2_samples.py \
        --phase2_dir  /path/to/phase2_data_dir \
        --out_dir     eval_results/phase2_inspect \
        --n_samples   20 \
        --seed        42 \
        --train_only          # filter_for_training 통과한 것만 검사
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse
import numpy as np
import torch

from dataset.phase2_dataset_builder import Phase2DatasetBuilder


# ─────────────────────────────────────────────────────────────────────────────
# 단순 CV2 없이도 동작하는 패널 합성
# ─────────────────────────────────────────────────────────────────────────────

def _text_row(W: int, text: str, h: int = 18) -> np.ndarray:
    """문자열을 간단한 흰글씨 검정배경 row로 반환 (cv2 의존)."""
    try:
        import cv2
        row = np.zeros((h, W, 3), dtype=np.uint8)
        cv2.putText(row, text[:140], (4, h - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 200, 200), 1, cv2.LINE_AA)
        return row
    except ImportError:
        return np.zeros((h, W, 3), dtype=np.uint8)


def _overlay_bbox(img: np.ndarray, bbox_norm, color=(0, 255, 0)) -> np.ndarray:
    """bbox_norm=[x1,y1,x2,y2] normalized → 픽셀 bbox 사각형 overlay."""
    if not bbox_norm or len(bbox_norm) < 4:
        return img
    try:
        import cv2
        H, W = img.shape[:2]
        x1 = int(bbox_norm[0] * W)
        y1 = int(bbox_norm[1] * H)
        x2 = int(bbox_norm[2] * W)
        y2 = int(bbox_norm[3] * H)
        out = img.copy()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        return out
    except ImportError:
        return img


def _overlay_mask(img: np.ndarray, mask: np.ndarray,
                  color=(0, 200, 255), alpha=0.45) -> np.ndarray:
    if mask is None or mask.sum() == 0:
        return img
    out = img.copy().astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + np.array(color) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _action_curve_strip(action_cond: np.ndarray, W: int, H: int = 48) -> np.ndarray:
    """action_cond (T, 7) → 각 dim별 곡선을 가로로 배치한 strip."""
    strip = np.zeros((H, W, 3), dtype=np.uint8)
    T, D = action_cond.shape
    try:
        import cv2
        xs = np.linspace(0, W - 1, T).astype(int)
        colors_d = [(255,100,100),(100,255,100),(100,100,255),
                    (255,255,100),(100,255,255),(255,100,255),(200,200,200)]
        for d in range(D):
            vals = action_cond[:, d]
            lo, hi = vals.min(), vals.max()
            span = hi - lo + 1e-8
            ys = ((1 - (vals - lo) / span) * (H - 4) + 2).astype(int)
            ys = np.clip(ys, 0, H - 1)
            c = colors_d[d % len(colors_d)]
            for j in range(len(xs) - 1):
                cv2.line(strip, (xs[j], ys[j]), (xs[j+1], ys[j+1]), c, 1)
    except ImportError:
        pass
    return strip


# ─────────────────────────────────────────────────────────────────────────────
# 단일 sample 시각화
# ─────────────────────────────────────────────────────────────────────────────

def inspect_sample(s: dict, root: str, out_dir: str, idx: int):
    """
    하나의 sample에 대해 contact sheet PNG를 out_dir에 저장.
    실패 시 warning만 출력하고 계속 진행.
    """
    sid = s['sample_id']

    def _load_frame(ref) -> np.ndarray | None:
        if ref is None:
            return None
        p = os.path.join(root, ref)
        if not os.path.exists(p):
            return None
        return np.load(p)

    def _load_latent(ref) -> torch.Tensor | None:
        if ref is None:
            return None
        p = os.path.join(root, ref)
        if not os.path.exists(p):
            return None
        return torch.load(p, map_location='cpu')

    def _load_mask(ref) -> np.ndarray | None:
        if ref is None:
            return None
        p = os.path.join(root, ref)
        if not os.path.exists(p):
            return None
        return np.load(p)

    bad_gen   = _load_frame(s.get('bad_generated_frame_ref'))
    gt_real   = _load_frame(s.get('gt_real_frame_ref'))
    last_good = _load_frame(s.get('last_good_frame_ref'))
    mask      = _load_mask(s.get('last_good_mask_ref'))

    if bad_gen is None or gt_real is None:
        print(f"  [WARN] {sid}: frame file missing, skip")
        return

    H, W = bad_gen.shape[:2]
    blank = np.full((H, W, 3), 60, dtype=np.uint8)

    # ── tracking_info bbox overlay ────────────────────────────────────
    tracking_info_path = os.path.join(root, sid, 'tracking_info.json')
    first_bbox = None
    if os.path.exists(tracking_info_path):
        with open(tracking_info_path) as f:
            ti = json.load(f)
        labels = ti.get('labels', {})
        for lbl, info in labels.items():
            bb = info.get('bbox_norm')
            if bb and len(bb) == 4 and any(v > 0 for v in bb):
                first_bbox = bb
                break

    bad_vis   = _overlay_bbox(bad_gen, first_bbox, (0, 255, 255))
    gt_vis    = _overlay_bbox(gt_real, first_bbox, (0, 255, 0))
    lg_vis    = (last_good.copy() if last_good is not None else blank.copy())
    if mask is not None and last_good is not None:
        lg_vis = _overlay_mask(lg_vis, mask)

    # ── action curve strip ────────────────────────────────────────────
    ac_path = os.path.join(root, s.get('action_window_ref', ''))
    action_strip = np.zeros((48, W * 3, 3), dtype=np.uint8)
    if os.path.exists(ac_path):
        ac = np.load(ac_path)
        action_strip = _action_curve_strip(ac, W * 3)

    # ── latent sanity (shape, min/max) ────────────────────────────────
    bg_lat = _load_latent(s.get('bad_generated_latent_ref'))
    gt_lat = _load_latent(s.get('gt_real_latent_ref'))
    lat_ok = (bg_lat is not None and gt_lat is not None
              and bg_lat.shape == gt_lat.shape)
    lat_info = (f"lat_shape={tuple(bg_lat.shape) if bg_lat is not None else '?'}"
                f"  match={'OK' if lat_ok else 'MISMATCH'}")

    # ── assemble contact sheet ────────────────────────────────────────
    try:
        import cv2
        panel_row = np.concatenate([bad_vis, gt_vis, lg_vis], axis=1)
        header = _text_row(W * 3,
            f"#{idx:03d}  {sid}  li={s.get('local_idx','?')}"
            f"  ri={s.get('real_video_frame_idx','?')}"
            f"  view={s.get('view_id','?')}  label={s.get('human_label','?')}"
            f"  tv={s.get('tracking_valid','?')}  {lat_info}",
            h=20)
        sub_header = _text_row(W * 3,
            f"  bad_gen [cyan bbox] | gt_real [green bbox] | last_good+mask", h=16)
        sheet = np.concatenate([header, sub_header, panel_row, action_strip], axis=0)
        fname = f"{idx:03d}_{sid}_li{s.get('local_idx','?')}_ri{s.get('real_video_frame_idx','?')}_v{s.get('view_id','?')}.png"
        cv2.imwrite(os.path.join(out_dir, fname), sheet[:, :, ::-1])
        print(f"  [{idx:03d}] saved: {fname}  lat={'OK' if lat_ok else '!MISMATCH'}")
    except ImportError:
        print(f"  [{idx:03d}] cv2 not available, skip image save")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase2_dir', type=str, required=True,
                        help='Phase2DatasetBuilder의 root 디렉토리')
    parser.add_argument('--out_dir',    type=str, default='eval_results/phase2_inspect')
    parser.add_argument('--n_samples',  type=int, default=20)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--train_only', action='store_true',
                        help='filter_for_training 통과한 sample만 검사')
    args = parser.parse_args()

    idx_path = os.path.join(args.phase2_dir, 'selected_samples.json')
    if not os.path.exists(idx_path):
        print(f"ERROR: {idx_path} not found")
        sys.exit(1)
    with open(idx_path) as f:
        samples = json.load(f)

    print(f"Total samples in index: {len(samples)}")

    if args.train_only:
        samples = Phase2DatasetBuilder.filter_for_training(samples)
        print(f"After filter_for_training: {len(samples)}")

    if len(samples) == 0:
        print("No samples to inspect.")
        return

    random.seed(args.seed)
    chosen = random.sample(samples, min(args.n_samples, len(samples)))
    print(f"Inspecting {len(chosen)} samples (seed={args.seed})")

    os.makedirs(args.out_dir, exist_ok=True)

    # skipped_no_gt 리포트
    skip_path = os.path.join(args.phase2_dir, 'skipped_no_gt.json')
    if os.path.exists(skip_path):
        with open(skip_path) as f:
            skipped = json.load(f)
        print(f"Skipped (no GT): {len(skipped)}")
        with open(os.path.join(args.out_dir, 'skipped_summary.json'), 'w') as f:
            json.dump(skipped[:50], f, indent=2)  # 최대 50개만

    # label 분포 요약
    from collections import Counter
    label_dist  = Counter(s.get('human_label')  for s in samples)
    src_dist    = Counter(s.get('label_source')  for s in samples)
    tv_dist     = Counter(s.get('tracking_valid') for s in samples)
    gt_dist     = Counter(s.get('gt_available')  for s in samples)

    # skipped reason 분포 (skipped_no_gt.json 전체 기준)
    skipped_reason_dist: dict = {}
    if os.path.exists(skip_path):
        with open(skip_path) as f:
            skipped_all = json.load(f)
        skipped_reason_dist = dict(Counter(s.get('reason') for s in skipped_all))

    summary = {
        'total': len(samples),
        'chosen': len(chosen),
        'human_label_distribution': dict(label_dist),
        'label_source_distribution': dict(src_dist),
        'tracking_valid_distribution': dict(tv_dist),
        'gt_available_distribution': dict(gt_dist),
        'filter_for_training_count': len(Phase2DatasetBuilder.filter_for_training(samples)),
        'skipped_reason_distribution': skipped_reason_dist,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {json.dumps(summary, indent=2)}")

    # 개별 contact sheet 저장
    for idx, s in enumerate(chosen):
        inspect_sample(s, args.phase2_dir, args.out_dir, idx)

    print(f"\nInspection complete → {args.out_dir}")
    print("Phase2 학습 시작 전 위 contact sheet를 확인하여 "
          "bad_gen/gt_real/last_good 매핑이 올바른지 검증하라.")


if __name__ == '__main__':
    main()
