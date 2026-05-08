"""
dataset/phase2_dataset_builder.py
Phase2 supervised training dataset builder.

설계 원칙
─────────
- 각 bad generated frame마다 동일 episode/start_idx/local_idx/view_id의
  GT real frame + GT real latent를 반드시 매칭.
- GT frame 또는 GT latent를 찾을 수 없으면 sample을 디렉토리째 저장하지 않고
  skipped_no_gt.json에만 기록.
- tracking_info.json의 bbox/shape_score/area_ratio/shape_latent는
  loss region selector와 negative descriptor로만 사용 (positive target 금지).
- bad generated frame / generated tracking shape_latent를 positive target으로 쓰지 않음.
- human_label 확정 조건: tracking_valid=True AND failure_detected=True AND gt_available=True
  → 'tracking_ok_generation_bad'. 나머지는 None (filter_for_training에서 제외).

Frame index mapping (overlapping-window rollout 기준)
───────────────────────────────────────────────────────
  rollout 각 step은 (pred_step-1) 프레임씩 advance (마지막 1 프레임 overlap).
  step_i에서 intra-step frame t의 loaded trajectory local index:

    local_idx = start_id + t
    start_id  = step_i * (pred_step - 1)     ← rollout_interact_pi_online.py의 start_id

  original video frame index (초기 로드 시 val_skip 간격):
    real_video_idx = start_idx + local_idx * val_skip

  주의: step_i * pred_step + t (비-overlap 공식)와 다름.
        rollout이 overlap window를 쓰므로 반드시 start_id를 caller에서 전달할 것.

디렉토리 구조
─────────────
  {root}/
    {sample_id}/
      bad_gen_frame.npy            (H, W, 3) uint8  ← input 또는 negative
      bad_gen_latent.pt            (4, h, w)         ← input 또는 negative  [view_id specific]
      gt_real_frame.npy            (H, W, 3) uint8  ← positive target
      gt_real_latent.pt            (4, h, w)         ← positive target      [same VAE, same scale]
      last_good_frame.npy          (H, W, 3) uint8  ← object consistency anchor
      last_good_crop_{label}.npy   (crop_H, crop_W, 3)
      last_good_mask_{label}.npy   (H, W) bool
      action_cond.npy              (num_history + pred_step, 7)
      history_latents.pt           (1, num_history, 4, h, w)
      current_latent.pt            (1, 4, h, w)
      contact_sheet_li{li}_ri{ri}_v{view_id}.png   ← sanity-check 이미지
      tracking_info.json           ← loss region selector + negative descriptor ONLY
      meta.json                    ← config snapshot + frame index + latent stats
    selected_samples.json
    skipped_no_gt.json
"""

import os
import json
import numpy as np
import torch
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Frame index mapping
# ─────────────────────────────────────────────────────────────────────────────

def local_frame_idx(start_id: int, t: int) -> int:
    """
    overlapping-window rollout:  local_idx = start_id + t
      start_id = step_i * (pred_step - 1)   (caller에서 계산해 전달)

    !! step_i * pred_step + t (비-overlap 공식)와 혼동 금지.
       rollout_interact_pi_online.py의 start_id = int(i * (pred_step - 1))이 맞다면
       이 함수가 올바른 local_idx를 반환한다.
    """
    return int(start_id) + int(t)


def real_video_frame_idx(start_id: int, t: int, start_idx: int, val_skip: int) -> int:
    """
    generated_frame[t] in rollout step with start_id
      → original video frame index  start_idx + local_idx * val_skip
    val_skip == frame_stride (e.g. 3 for 15Hz annotation → 5Hz video)
    """
    li = local_frame_idx(start_id, t)
    return start_idx + li * val_skip


# ─────────────────────────────────────────────────────────────────────────────
# Phase2DatasetBuilder
# ─────────────────────────────────────────────────────────────────────────────

class Phase2DatasetBuilder:
    """
    Phase2 supervised training dataset builder.

    [학습 포함 조건]
      human_label == 'tracking_ok_generation_bad'
      AND gt_available == True
      AND tracking_valid == True
      AND gt_real_latent_ref is not None
      AND bad_generated_latent_ref is not None
      AND action_window_ref is not None

    [human_label 확정 로직]
      tracking_valid=True AND failure_detected=True AND gt_available=True
        → human_label = 'tracking_ok_generation_bad'
      그 외 → human_label = None  (추후 human annotation 또는 자동 labeler 대기)

    [tracking_info.json 사용 범위]
      bbox / shape_score / area_ratio / shape_latent:
        ONLY for loss region selector and negative descriptor.
        절대 positive training target으로 사용하지 말 것.
    """

    TRAIN_LABEL = 'tracking_ok_generation_bad'

    def __init__(self, root: str, config_snapshot: Dict[str, Any]):
        """
        config_snapshot 필수 키:
          val_id, start_idx, pred_step, fps, down_sample, val_skip, text, view_id
        """
        self.root = root
        self.cfg  = config_snapshot
        os.makedirs(root, exist_ok=True)
        self._samples:  List[Dict] = []
        self._skipped:  List[Dict] = []
        # 기존 인덱스가 있으면 이어서 누적
        _idx = os.path.join(root, 'selected_samples.json')
        if os.path.exists(_idx):
            with open(_idx) as f:
                self._samples = json.load(f)
        _skip_idx = os.path.join(root, 'skipped_no_gt.json')
        if os.path.exists(_skip_idx):
            with open(_skip_idx) as f:
                self._skipped = json.load(f)

    # ── main sample saver ─────────────────────────────────────────────────

    def save_failure_sample(
        self,
        sample_id:             str,
        step_i:                int,
        start_id:              int,          # = step_i * (pred_step-1), from rollout
        bad_t:                 int,          # intra-step frame index of first bad frame
        view_id:               int,
        bad_gen_frame:         np.ndarray,   # (H, W, 3) uint8, view_id
        bad_gen_latent:        torch.Tensor, # (4, h, w), view_id ONLY (not 3-view stacked)
        video_dict:            list,         # list[view] → (N, H, W, 3) uint8
        video_latents:         list,         # list[view] → (N, 4, h, w) Tensor [same VAE]
        registry,                            # ObjectRegistry
        trigger_labels:        list,
        action_cond:           np.ndarray,   # (num_history + pred_step, 7)
        history_latents:       torch.Tensor, # (1, num_history, 4, h, w)
        current_latent:        torch.Tensor, # (1, 4, h, w)
        failure_group:         str,
        failure_causes:        dict,
        object_labels:         list,
        tracking_objects_info: dict,         # {label: {bbox_norm, shape_score, ...}}
        tracking_valid:        bool,         # SAM3 tracking was reliable before failure
        failure_detected:      bool,         # neg_event_detected == True
    ) -> Optional[Dict]:
        """
        Save one failure sample with its GT counterpart.

        GT를 찾지 못하면 디렉토리를 만들지 않고 skipped_no_gt에 기록한 뒤 None 반환.
        latent shape 불일치 시에도 sample을 폐기한다.
        """
        cfg = self.cfg
        li  = local_frame_idx(start_id, bad_t)
        ri  = real_video_frame_idx(start_id, bad_t, cfg['start_idx'], cfg['val_skip'])

        # ── GT availability check ──────────────────────────────────────
        n_frames     = video_dict[view_id].shape[0]
        gt_available = (li < n_frames)

        if not gt_available:
            detail = f"local_idx={li} >= n_loaded_frames={n_frames}"
            print(f"[Phase2] SKIP {sample_id}: GT not available ({detail})")
            self._skipped.append({
                'sample_id': sample_id, 'reason': 'no_gt', 'detail': detail,
                'step_i': step_i, 'bad_t': bad_t, 'local_idx': li,
            })
            self._flush_skipped()
            return None

        gt_frame  = video_dict[view_id][li]           # (H, W, 3) uint8
        gt_latent = video_latents[view_id][li].cpu()  # (4, h, w)

        # ── latent shape / dtype sanity check ─────────────────────────
        bg_lat  = bad_gen_latent.cpu()
        if bg_lat.shape != gt_latent.shape:
            detail = (f"bad_gen={tuple(bg_lat.shape)} gt={tuple(gt_latent.shape)}")
            print(f"[Phase2] DISCARD {sample_id}: latent shape mismatch {detail}")
            self._skipped.append({
                'sample_id': sample_id, 'reason': 'latent_shape_mismatch', 'detail': detail,
                'step_i': step_i, 'bad_t': bad_t, 'local_idx': li,
            })
            self._flush_skipped()
            return None

        # ── frame shape sanity check ───────────────────────────────────
        if bad_gen_frame.shape != gt_frame.shape:
            detail = (f"bad_gen={bad_gen_frame.shape} gt={gt_frame.shape}")
            print(f"[Phase2] DISCARD {sample_id}: frame shape mismatch {detail}")
            self._skipped.append({
                'sample_id': sample_id, 'reason': 'frame_shape_mismatch', 'detail': detail,
                'step_i': step_i, 'bad_t': bad_t, 'local_idx': li,
            })
            self._flush_skipped()
            return None

        # ── nan/inf latent check ───────────────────────────────────────
        bg_f, gt_f = bg_lat.float(), gt_latent.float()
        bg_nan = bool(torch.isnan(bg_f).any())
        bg_inf = bool(torch.isinf(bg_f).any())
        gt_nan = bool(torch.isnan(gt_f).any())
        gt_inf = bool(torch.isinf(gt_f).any())
        if bg_nan or bg_inf or gt_nan or gt_inf:
            detail = (f"bg: nan={bg_nan} inf={bg_inf}  gt: nan={gt_nan} inf={gt_inf}")
            print(f"[Phase2] DISCARD {sample_id}: nan_or_inf_latent  {detail}")
            self._skipped.append({
                'sample_id': sample_id, 'reason': 'nan_or_inf_latent', 'detail': detail,
                'step_i': step_i, 'bad_t': bad_t, 'local_idx': li,
            })
            self._flush_skipped()
            return None

        # ── latent stats (for meta.json) ───────────────────────────────
        bg_stats = self._latent_stats(bg_lat)
        gt_stats = self._latent_stats(gt_latent)

        # ── human_label 확정 ───────────────────────────────────────────
        # auto_label_candidate: 조건이 충족되는지 표시 (human이 override 가능)
        # human_label 확정: tracking_valid AND failure_detected AND gt_available
        auto_label_candidate = (self.TRAIN_LABEL
                                if (tracking_valid and failure_detected and gt_available)
                                else None)
        label_source = 'auto_rule' if auto_label_candidate is not None else None
        human_label  = auto_label_candidate  # human annotation 전까지 candidate 사용

        # ── 디렉토리 생성 및 파일 저장 ─────────────────────────────────
        sdir = os.path.join(self.root, sample_id)
        os.makedirs(sdir, exist_ok=True)
        refs = {}

        # bad generated frame + latent (input / negative)
        np.save(os.path.join(sdir, 'bad_gen_frame.npy'), bad_gen_frame)
        refs['bad_generated_frame_ref']  = os.path.join(sample_id, 'bad_gen_frame.npy')
        torch.save(bg_lat, os.path.join(sdir, 'bad_gen_latent.pt'))
        refs['bad_generated_latent_ref'] = os.path.join(sample_id, 'bad_gen_latent.pt')

        # GT real frame + latent (positive target)
        np.save(os.path.join(sdir, 'gt_real_frame.npy'), gt_frame)
        refs['gt_real_frame_ref']  = os.path.join(sample_id, 'gt_real_frame.npy')
        torch.save(gt_latent, os.path.join(sdir, 'gt_real_latent.pt'))
        refs['gt_real_latent_ref'] = os.path.join(sample_id, 'gt_real_latent.pt')

        # last_good frame / crop / mask (object consistency anchor)
        refs['last_good_frame_ref'] = None
        refs['last_good_crop_ref']  = None
        refs['last_good_mask_ref']  = None
        _frame_saved = False
        last_good_frame_arr = None
        for lbl in trigger_labels:
            obj = registry.get(lbl) if registry is not None else None
            if obj is None:
                continue
            safe_lbl = lbl.replace(' ', '_').replace('/', '-')
            if obj.last_good_frame is not None and not _frame_saved:
                last_good_frame_arr = obj.last_good_frame
                np.save(os.path.join(sdir, 'last_good_frame.npy'), last_good_frame_arr)
                refs['last_good_frame_ref'] = os.path.join(sample_id, 'last_good_frame.npy')
                _frame_saved = True
            if obj.last_good_mask is not None and obj.last_good_frame is not None:
                np.save(os.path.join(sdir, f'last_good_mask_{safe_lbl}.npy'),
                        obj.last_good_mask)
                refs[f'last_good_mask_ref_{safe_lbl}'] = os.path.join(
                    sample_id, f'last_good_mask_{safe_lbl}.npy')
                ys, xs = np.where(obj.last_good_mask)
                if len(ys) > 0:
                    y1, y2 = int(ys.min()), int(ys.max()) + 1
                    x1, x2 = int(xs.min()), int(xs.max()) + 1
                    crop = obj.last_good_frame[y1:y2, x1:x2]
                    np.save(os.path.join(sdir, f'last_good_crop_{safe_lbl}.npy'), crop)
                    refs[f'last_good_crop_ref_{safe_lbl}'] = os.path.join(
                        sample_id, f'last_good_crop_{safe_lbl}.npy')
        first_safe = (trigger_labels[0].replace(' ', '_').replace('/', '-')
                      if trigger_labels else None)
        if first_safe:
            refs['last_good_crop_ref'] = refs.get(f'last_good_crop_ref_{first_safe}')
            refs['last_good_mask_ref'] = refs.get(f'last_good_mask_ref_{first_safe}')

        # action_cond — same timestep as bad frame
        # action_window_start: local_idx of history start (depends on history_idx)
        # For simplicity, record the slice that corresponds to the prediction window
        pred_start_li  = li - int(cfg['pred_step']) + 1   # earliest frame in pred window
        action_pred_end_li = li
        np.save(os.path.join(sdir, 'action_cond.npy'), action_cond)
        refs['action_window_ref'] = os.path.join(sample_id, 'action_cond.npy')

        # history latents + current latent
        torch.save(history_latents.cpu(), os.path.join(sdir, 'history_latents.pt'))
        torch.save(current_latent.cpu(),  os.path.join(sdir, 'current_latent.pt'))

        # contact sheet: bad_gen | gt_real | last_good  (sanity check)
        contact_name = (f"contact_li{li}_ri{ri}_v{view_id}.png")
        _save_contact_sheet(
            bad_gen_frame, gt_frame, last_good_frame_arr,
            os.path.join(sdir, contact_name),
            title=f"{sample_id}  li={li} ri={ri} view={view_id}",
        )
        refs['contact_sheet_ref'] = os.path.join(sample_id, contact_name)

        # tracking_info.json (loss region selector + negative descriptor ONLY)
        tracking_info = {
            'USAGE_NOTE': (
                'bbox / shape_score / area_ratio / shape_latent are for '
                'loss_region_selector and negative_descriptor ONLY. '
                'Do NOT use as positive training target.'
            ),
            'labels': {
                lbl: {
                    'bbox_norm':    tracking_objects_info.get(lbl, {}).get('bbox_norm', []),
                    'shape_score':  tracking_objects_info.get(lbl, {}).get('shape_score'),
                    'area_ratio':   tracking_objects_info.get(lbl, {}).get('area_ratio'),
                    'absent':       tracking_objects_info.get(lbl, {}).get('absent', False),
                    'shape_latent': tracking_objects_info.get(lbl, {}).get('shape_latent', []),
                }
                for lbl in object_labels
            },
        }
        with open(os.path.join(sdir, 'tracking_info.json'), 'w') as f:
            json.dump(tracking_info, f, indent=2)

        # meta.json (config snapshot + frame index mapping + latent stats)
        meta = {
            'sample_id':        sample_id,
            'traj_id':          cfg['val_id'],
            # frame index mapping — full audit trail
            'step_i':           step_i,
            'intra_step_t':     bad_t,
            'start_id':         start_id,        # = step_i * (pred_step - 1)
            'local_idx':        li,              # = start_id + bad_t
            'start_idx':        cfg['start_idx'],
            'val_skip':         cfg['val_skip'],
            'frame_stride':     cfg['val_skip'],
            'real_video_frame_idx': ri,          # = start_idx + local_idx * val_skip
            'pred_step':        cfg['pred_step'],
            'fps':              cfg['fps'],
            'down_sample':      cfg['down_sample'],
            'view_id':          view_id,
            # action window
            'action_cond_shape':         list(action_cond.shape),
            'action_pred_start_local_idx': pred_start_li,
            'action_pred_end_local_idx':   action_pred_end_li,
            # latent stats (sanity check)
            'bad_gen_latent_stats': bg_stats,
            'gt_real_latent_stats': gt_stats,
            # labels + validity
            'object_labels':    object_labels,
            'text':             cfg['text'],
            'trigger_labels':   trigger_labels,
            'failure_group':    failure_group,
            'failure_causes':   failure_causes,
            'tracking_valid':   tracking_valid,
            'failure_detected': failure_detected,
            'gt_available':     gt_available,
            'auto_label_candidate': auto_label_candidate,
            'label_source':     label_source,
            'human_label':      human_label,
        }
        with open(os.path.join(sdir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # selected_samples.json entry
        entry = {
            'sample_id':               sample_id,
            'traj_id':                 cfg['val_id'],
            'view_id':                 view_id,
            'step_i':                  step_i,
            'intra_step_t':            bad_t,
            'local_idx':               li,
            'real_video_frame_idx':    ri,
            'frame_stride':            cfg['val_skip'],
            'start_idx':               cfg['start_idx'],
            'pred_step':               cfg['pred_step'],
            'fps':                     cfg['fps'],
            'down_sample':             cfg['down_sample'],
            'val_skip':                cfg['val_skip'],
            'bad_generated_frame_ref': refs['bad_generated_frame_ref'],
            'bad_generated_latent_ref': refs['bad_generated_latent_ref'],
            'gt_real_frame_ref':       refs['gt_real_frame_ref'],
            'gt_real_latent_ref':      refs['gt_real_latent_ref'],
            'last_good_frame_ref':     refs.get('last_good_frame_ref'),
            'last_good_crop_ref':      refs.get('last_good_crop_ref'),
            'last_good_mask_ref':      refs.get('last_good_mask_ref'),
            'action_window_ref':       refs['action_window_ref'],
            'action_pred_start_local_idx': pred_start_li,
            'action_pred_end_local_idx':   action_pred_end_li,
            'contact_sheet_ref':       refs['contact_sheet_ref'],
            'auto_label_candidate':    auto_label_candidate,
            'label_source':            label_source,
            'human_label':             human_label,
            'failure_type':            failure_group,
            'gt_available':            True,        # already verified above
            'tracking_valid':          tracking_valid,
            'failure_detected':        failure_detected,
            'trigger_labels':          trigger_labels,
            'failure_causes':          failure_causes,
        }
        self._samples.append(entry)
        self.flush()

        status = ('TRAIN_CANDIDATE' if human_label == self.TRAIN_LABEL
                  else 'human_label=None (pending annotation)')
        print(f"[Phase2] SAVED {sample_id}  li={li} ri={ri} view={view_id}"
              f"  tracking_valid={tracking_valid}  label={human_label}  → {status}")
        return entry

    # ── flush ─────────────────────────────────────────────────────────────

    def flush(self):
        with open(os.path.join(self.root, 'selected_samples.json'), 'w') as f:
            json.dump(self._samples, f, indent=2)

    def _flush_skipped(self):
        with open(os.path.join(self.root, 'skipped_no_gt.json'), 'w') as f:
            json.dump(self._skipped, f, indent=2)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _latent_stats(t: torch.Tensor) -> dict:
        f = t.float()
        return {
            'shape':   list(t.shape),
            'dtype':   str(t.dtype),
            'min':     float(f.min()),
            'max':     float(f.max()),
            'mean':    float(f.mean()),
            'std':     float(f.std()),
            'has_nan': bool(torch.isnan(f).any()),
            'has_inf': bool(torch.isinf(f).any()),
        }

    @staticmethod
    def compute_tracking_valid(
        trigger_labels:        list,
        failure_group:         str,
        tracking_objects_info: dict,
        shape_score_threshold: float = 0.75,
        area_ratio_min:        float = 0.1,
        area_ratio_max:        float = 10.0,
    ) -> bool:
        """
        SAM3 tracking이 신뢰할 수 있는지 종합 판정.

        조건 (AND):
          1. trigger_labels 비어있지 않음
          2. failure_group != 'uncertain'
          3. 각 trigger label에 대해:
             - absent == False
             - shape_rejected == False
             - bbox_norm이 유효 (non-zero 원소 존재)
             - shape_score >= shape_score_threshold
             - area_ratio in [area_ratio_min, area_ratio_max]
        """
        if not trigger_labels or failure_group == 'uncertain':
            return False
        for lbl in trigger_labels:
            info = tracking_objects_info.get(lbl, {})
            if info.get('absent', True):
                return False
            if info.get('shape_rejected', True):
                return False
            bbox = info.get('bbox_norm', [])
            if not (len(bbox) >= 4 and any(v > 0 for v in bbox)):
                return False
            ss = info.get('shape_score')
            if ss is None or ss < shape_score_threshold:
                return False
            ar = info.get('area_ratio')
            if ar is None or not (area_ratio_min <= ar <= area_ratio_max):
                return False
        return True

    # ── training filter ───────────────────────────────────────────────────

    @staticmethod
    def filter_for_training(samples: list) -> list:
        """
        Phase2 supervised training에 포함할 sample만 반환.

        조건 (AND):
          human_label  == 'tracking_ok_generation_bad'
          gt_available == True
          tracking_valid == True
          gt_real_latent_ref   is not None
          bad_generated_latent_ref is not None
          action_window_ref    is not None

        제외: tracking_bad_generation_unknown, tracking_lost, uncertain, human_label=None,
              GT 없음, tracking_valid=False
        """
        LABEL = Phase2DatasetBuilder.TRAIN_LABEL
        return [
            s for s in samples
            if (s.get('human_label')             == LABEL
                and s.get('gt_available',    False)
                and s.get('tracking_valid',  False)
                and s.get('gt_real_latent_ref')   is not None
                and s.get('bad_generated_latent_ref') is not None
                and s.get('action_window_ref')    is not None)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Contact sheet helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_contact_sheet(
    bad_gen: np.ndarray,
    gt_real: np.ndarray,
    last_good: Optional[np.ndarray],
    path: str,
    title: str = "",
):
    """
    bad_gen | gt_real | last_good (없으면 회색 blank) 를 가로로 이어붙인 PNG 저장.
    각 패널 위에 label 텍스트를 추가한다.
    """
    try:
        import cv2
    except ImportError:
        return

    H, W = bad_gen.shape[:2]
    blank = np.full((H, W, 3), 128, dtype=np.uint8)

    def _label(img, text):
        out = img.copy()
        cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 0), 1, cv2.LINE_AA)
        return out

    panels = [
        _label(bad_gen, "bad_gen"),
        _label(gt_real, "gt_real"),
        _label(last_good if last_good is not None else blank, "last_good"),
    ]
    sheet = np.concatenate(panels, axis=1)

    # title strip
    strip = np.zeros((20, sheet.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, title[:120], (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (200, 200, 200), 1, cv2.LINE_AA)
    sheet = np.concatenate([strip, sheet], axis=0)

    cv2.imwrite(path, sheet[:, :, ::-1])  # RGB→BGR
