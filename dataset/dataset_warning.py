"""
Warning-conditioned Ctrl-World 학습용 데이터셋.

기대 디렉토리 구조:
  tracking_root/
    ep0000_Put_the_marker_in_the_pot/
      tracking.json           ← SAM3 object tracking annotation
      latent.pt               ← (T, 4, H, W) pre-encoded VAE latents (3-view stacked)

tracking.json 포맷:
{
  "episode_id": "ep0000_...",
  "object_labels": ["pen", "pot"],
  "frames": [
    {
      "frame_idx": 0,
      "action": [x, y, z, rx, ry, rz, g],
      "objects": {
        "pen": {
          "absent":        false,
          "cause":         null,
          "bad_streak":    0,
          "error_score":   0.0,
          "iou":           null,
          "state":         0.0,
          "shape_score":   1.0,
          "shape_rejected": false,
          "area_ratio":    1.0,
          "extent_ratio":  1.0,
          "bbox":          [0.1, 0.2, 0.3, 0.4],
          "appearance":    [...],   // 512-dim float list
          "shape_latent":  [...]    // 256-dim float list
        }
      }
    }
  ]
}

각 샘플: (history_len=6 입력 프레임, pred_len=1 타깃 프레임)
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from models.object_registry import FEATURE_DIM, MAX_OBJECTS, SHAPE_SIZE
from models.warning_utils import (
    process_episode_warnings,
    HARD_HORIZON, SOFT_PERSISTENCE, WARNING_DIM,
)

HISTORY_LEN = 6
PRED_LEN    = 1


class WarningDataset(Dataset):
    """
    mode: 'baseline' | 'obj' | 'hard' | 'soft' | 'full'
      baseline : obj_state/shape/warning 모두 zeros
      obj      : obj_state + obj_shape, warning zeros
      hard     : obj + hard warning only  (soft_any / soft_* 를 0으로 마스킹)
      soft     : obj + soft warning only  (hard_any / hard_* 를 0으로 마스킹)
      full     : obj + hard + soft
    """

    def __init__(self, tracking_root: str, mode: str = 'full',
                 history_len: int = HISTORY_LEN, pred_len: int = PRED_LEN):
        super().__init__()
        assert mode in ('baseline', 'obj', 'hard', 'soft', 'full')
        self.tracking_root = tracking_root
        self.mode          = mode
        self.history_len   = history_len
        self.pred_len      = pred_len

        self.samples = []   # list of (ep_dir, start_frame_idx)
        self._load_episodes()

    def _load_episodes(self):
        for ep_name in sorted(os.listdir(self.tracking_root)):
            ep_dir      = os.path.join(self.tracking_root, ep_name)
            track_path  = os.path.join(ep_dir, 'tracking.json')
            latent_path = os.path.join(ep_dir, 'latent.pt')
            if not (os.path.isfile(track_path) and os.path.isfile(latent_path)):
                continue

            with open(track_path) as f:
                meta = json.load(f)
            # action 필드가 없는 에피소드는 건너뜀
            if 'action' not in meta['frames'][0]:
                print(f"  [SKIP] {ep_name}: 'action' field missing in tracking.json")
                continue
            n_frames = len(meta['frames'])
            win = self.history_len + self.pred_len
            if n_frames < win:
                continue
            for start in range(n_frames - win + 1):
                self.samples.append((ep_dir, start, meta))

        print(f"[WarningDataset/{self.mode}] {len(self.samples)} samples from "
              f"{self.tracking_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_dir, start, meta = self.samples[idx]
        win      = self.history_len + self.pred_len
        frame_slice = meta['frames'][start: start + win]

        # ── 1) latent ────────────────────────────────────────
        latent_full = torch.load(os.path.join(ep_dir, 'latent.pt'),
                                 map_location='cpu')   # (T, 4, H, W)
        latent = latent_full[start: start + win].float()  # (win, 4, H, W)

        # ── 2) action ────────────────────────────────────────
        actions = np.array([f['action'] for f in frame_slice], dtype=np.float32)
        action  = torch.tensor(actions)   # (win, 7)

        # ── 3) text ──────────────────────────────────────────
        text = meta.get('language_instruction', '')

        # ── 4) obj_state / obj_shape (current = last history frame) ──
        cur_frame  = frame_slice[self.history_len - 1]  # 마지막 history 프레임
        obj_state, obj_shape = self._extract_obj_tensors(cur_frame, meta['object_labels'])

        # ── 5) warning_vec ────────────────────────────────────
        # 에피소드 전체 warning 시퀀스를 계산해서 현재 window의 current 시점 값 사용
        all_records = [
            [f['objects'].get(lbl, {}) for lbl in meta['object_labels']]
            for f in meta['frames']
        ]
        all_vecs = process_episode_warnings(all_records, HARD_HORIZON, SOFT_PERSISTENCE)
        cur_vec  = all_vecs[start + self.history_len - 1]   # (8,)
        warning_vec = self._apply_mode_mask(torch.tensor(cur_vec))

        if self.mode == 'baseline':
            obj_state   = torch.zeros_like(obj_state)
            obj_shape   = torch.zeros_like(obj_shape)

        return {
            'latent':      latent,         # (win, 4, H, W)
            'action':      action,         # (win, 7)
            'text':        text,
            'obj_state':   obj_state,      # (MAX_OBJECTS, 518)
            'obj_shape':   obj_shape,      # (MAX_OBJECTS, 256)
            'warning_vec': warning_vec,    # (8,)
        }

    # ── 헬퍼 ─────────────────────────────────────────────────
    def _extract_obj_tensors(self, frame: dict, labels: list):
        """frame dict → (MAX_OBJECTS, FEATURE_DIM), (MAX_OBJECTS, SHAPE_SIZE^2)"""
        feat_rows  = []
        shape_rows = []
        for lbl in labels[:MAX_OBJECTS]:
            obj = frame['objects'].get(lbl, {})
            presence = float(not obj.get('absent', True))
            bbox     = np.array(obj.get('bbox', [0.0] * 4), dtype=np.float32)
            state    = float(obj.get('state', 0.0))
            feat_rows.append(np.concatenate([
                [presence], bbox, [state]
            ]))  # 6-dim  (appearance 제외)

            shape_lat = np.array(obj.get('shape_latent',
                                         [0.0] * (SHAPE_SIZE * SHAPE_SIZE)),
                                 dtype=np.float32)
            shape_rows.append(shape_lat)

        n = len(feat_rows)
        obj_state = np.zeros((MAX_OBJECTS, FEATURE_DIM), dtype=np.float32)
        obj_shape = np.zeros((MAX_OBJECTS, SHAPE_SIZE * SHAPE_SIZE), dtype=np.float32)
        if n > 0:
            obj_state[:n] = np.stack(feat_rows)
            obj_shape[:n] = np.stack(shape_rows)

        return torch.tensor(obj_state), torch.tensor(obj_shape)

    def _apply_mode_mask(self, vec: torch.Tensor) -> torch.Tensor:
        """mode에 따라 warning_vec의 hard/soft 필드를 선택적으로 0으로 마스킹."""
        vec = vec.clone()
        if self.mode in ('baseline', 'obj'):
            vec[:] = 0.0
        elif self.mode == 'hard':
            # soft_any(1), soft_count(3), soft_max(5), soft_sum(7) 마스킹
            vec[[1, 3, 5, 7]] = 0.0
        elif self.mode == 'soft':
            # hard_any(0), hard_count(2), hard_max(4), hard_sum(6) 마스킹
            vec[[0, 2, 4, 6]] = 0.0
        # 'full': 마스킹 없음
        return vec
