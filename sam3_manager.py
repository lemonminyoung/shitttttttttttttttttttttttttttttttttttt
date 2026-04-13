# sam3_manager.py
import os
import tempfile
import shutil
import numpy as np
import torch
from PIL import Image


class SAM3Manager:
    """
    SAM3 video predictor 기반 멀티오브젝트 tracker.

    멀티오브젝트 등록 전략:
      - 단일 세션 + skip_reset=True 로 객체를 순차 등록
      - SAM3 내부 _get_visual_prompt에 force_new=True 패치 적용으로
        previous_stages_out 여부와 무관하게 두 번째 box도 새 객체로 탐지

    슬라이딩 윈도우:
      - 세션 구성: [anchor(실제 첫 프레임)] + [최근 window_size개 생성 프레임] + [새 chunk]

    Anchor box:
      - initialize()에서 real frame 탐지로 고정, 이후 변경 없음
      - 매 chunk 세션 frame_idx=0 box prompt로 사용

    Negative detection:
      - chunk 내 프레임별로 판정
      - 첫 bad frame 인덱스(first_bad_t)를 결과에 포함
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda", window_size: int = 4):
        from sam3.model_builder import build_sam3_video_predictor
        self.video_predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path)
        print("sam3 model dtype:", next(self.video_predictor.model.parameters()).dtype)
        self.window_size = window_size

        self.object_names = []
        self.object_masks = {}       # {name: (H, W) bool}
        self.obj_id_map = {}         # {name: int} — SAM3 내부 obj_id
        self.mask_area_history = {}  # {name: [area, ...]}
        self._anchor_boxes = {}      # {name: [x1,y1,x2,y2]} — initialize()에서 고정
        self.last_good_boxes = {}    # {name: [x1,y1,x2,y2]} — absence 판정용
        self._tmpdir = None
        self._anchor_frame = None
        self._recent_frames = []
        self._frame_hw = None

    # ──────────────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────────────
    def initialize(self, frame: np.ndarray, object_names: list):
        """
        frame: (H, W, 3) uint8
        object_names: ["robot arm and end-effector", "cup", ...]
        """
        self.object_names = object_names
        self.object_masks = {}
        self.obj_id_map = {}
        self.mask_area_history = {name: [] for name in object_names}
        self._anchor_frame = frame
        self._recent_frames = []
        self._frame_hw = frame.shape[:2]

        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = tempfile.mkdtemp()
        Image.fromarray(frame).save(os.path.join(self._tmpdir, "00000.jpg"))

        # 객체마다 독립 세션 + text prompt → 각자 anchor box 탐지
        for i, name in enumerate(object_names):
            sid = self.video_predictor.start_session(self._tmpdir)["session_id"]
            print(f"[INIT] name={name}")
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    r = self.video_predictor.add_prompt(
                        sid, frame_idx=0, text=name
                    )
                out = r["outputs"]
                masks = out.get("out_binary_masks", [])
                boxes = out.get("out_boxes_xywh")
                obj_ids = list(out.get("out_obj_ids", []))
                print(f"[INIT] {name}: out_obj_ids={obj_ids} mask_count={len(masks)}")

                if len(obj_ids) > 0:
                    self.obj_id_map[name] = int(obj_ids[-1])

                if len(masks) > 0:
                    m = np.array(masks[-1]).astype(bool)
                    self.object_masks[name] = m
                    if boxes is not None and len(boxes) > 0:
                        x, y, w, h = boxes[-1]
                        box = [x, y, x + w, y + h]
                    else:
                        H, W = self._frame_hw
                        ys, xs = np.where(m)
                        box = [xs.min()/W, ys.min()/H, xs.max()/W, ys.max()/H]
                    self._anchor_boxes[name] = box
                    self.last_good_boxes[name] = list(box)
                else:
                    print(f"[SAM3] '{name}' not found in initial frame")
                    self.object_masks[name] = None
                    self._anchor_boxes[name] = None
                    self.last_good_boxes[name] = None
            finally:
                try:
                    self.video_predictor.close_session(sid)
                except Exception:
                    pass

            area = float(self.object_masks[name].sum()) if self.object_masks[name] is not None else 0.0
            self.mask_area_history[name].append(area)

        print(f"[SAM3] Initialized: {object_names}")
        for name in object_names:
            mask = self.object_masks.get(name)
            print(f"  {name}: obj_id={self.obj_id_map.get(name)} "
                  f"area={mask.sum() if mask is not None else 0} "
                  f"box={self.last_good_boxes[name]}")

    # ──────────────────────────────────────────────────────
    # chunk 단위 업데이트
    # ──────────────────────────────────────────────────────
    def update_chunk(self, frames: list) -> list:
        """
        frames: list of (H, W, 3) uint8 — 3장 고정
        returns: list of dict {name: {"mask", "area", "absent", "cause", "first_bad_t"}}
        """
        assert self.object_names, "initialize() 먼저 호출해야 함"

        recent = self._recent_frames[-self.window_size:] if self.window_size > 0 else []
        session_frames = [self._anchor_frame] + recent + frames
        n_anchor = 1 + len(recent)
        print(f"[SAM3] 세션: anchor(1) + recent({len(recent)}) + new({len(frames)}) = {len(session_frames)}")

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = tempfile.mkdtemp()
        for i, f in enumerate(session_frames):
            Image.fromarray(f).save(os.path.join(self._tmpdir, f"{i:05d}.jpg"))

        # 객체별 독립 세션으로 propagate 후 merge
        merged_masks = {}
        per_obj_boxes = {name: {} for name in self.object_names}
        for name in self.object_names:
            obj_masks, obj_boxes = self._propagate_single_object(name, session_frames)
            for fidx, mask in obj_masks.items():
                merged_masks.setdefault(fidx, {})[name] = mask
            per_obj_boxes[name] = obj_boxes

        # last_good_boxes 갱신
        for name in self.object_names:
            boxes = per_obj_boxes.get(name, {})
            if boxes:
                self.last_good_boxes[name] = boxes[max(boxes.keys())]

        # chunk 첫 프레임 area baseline 교체
        first_chunk_masks = merged_masks.get(n_anchor, {})
        for name in self.object_names:
            m = first_chunk_masks.get(name)
            area = float(m.sum()) if m is not None else 0.0
            if self.mask_area_history[name]:
                self.mask_area_history[name][-1] = area

        # chunk 프레임별 absence 판정
        results_list = []
        first_bad_t = None

        for t, frame in enumerate(frames):
            fidx = n_anchor + t
            frame_masks = merged_masks.get(fidx, {})
            frame_results = {}

            for name in self.object_names:
                mask = frame_masks.get(name)
                area = float(mask.sum()) if mask is not None else 0.0
                self.mask_area_history[name].append(area)
                self.object_masks[name] = mask

                if t == 0:
                    absent, cause = False, None
                else:
                    absent, cause = self._check_absence(name, mask, frame)

                if absent and first_bad_t is None:
                    first_bad_t = t
                    print(f"[SAM3] first_bad_t={t}, '{name}' absent: {cause}")

                frame_results[name] = {
                    "mask":        mask,
                    "area":        area,
                    "absent":      absent,
                    "cause":       cause,
                    "first_bad_t": first_bad_t,
                }

            results_list.append(frame_results)

        self._recent_frames.extend(frames)
        if len(self._recent_frames) > self.window_size:
            self._recent_frames = self._recent_frames[-self.window_size:]

        return results_list

    # ──────────────────────────────────────────────────────
    # 단일 객체 propagation (독립 세션)
    # ──────────────────────────────────────────────────────
    def _propagate_single_object(self, name: str, session_frames: list):
        """
        단일 객체에 대해 독립 세션 → anchor box prompt → propagate.
        returns:
          masks: {frame_idx: (H,W) bool}
          boxes: {frame_idx: [x1,y1,x2,y2]}
        """
        box = self._anchor_boxes.get(name)
        sid = self.video_predictor.start_session(self._tmpdir)["session_id"]
        print(f"[RE-ADD] name={name}, box={box}")

        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if box is not None:
                    x1, y1, x2, y2 = box
                    r = self.video_predictor.add_prompt(
                        sid, frame_idx=0,
                        bounding_boxes=[[x1, y1, x2 - x1, y2 - y1]],
                        bounding_box_labels=[1],
                    )
                else:
                    r = self.video_predictor.add_prompt(sid, frame_idx=0, text=name)

            out_ids = r["outputs"].get("out_obj_ids", [])
            print(f"[RE-ADD] {name} → out_obj_ids={list(out_ids)} "
                  f"mask_count={len(r['outputs'].get('out_binary_masks', []))}")

            masks = {}
            boxes = {}
            first_logged = False
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for result in self.video_predictor.propagate_in_video(
                    sid, propagation_direction="forward"
                ):
                    fidx = result["frame_index"]
                    out  = result["outputs"]
                    if not first_logged:
                        print(f"[PROP/{name}] fidx={fidx}, "
                              f"out_obj_ids={list(out.get('out_obj_ids', []))}")
                        first_logged = True

                    obj_ids   = out.get("out_obj_ids", [])
                    out_masks = out.get("out_binary_masks", [])
                    out_boxes = out.get("out_boxes_xywh")

                    if len(obj_ids) > 0 and len(out_masks) > 0:
                        m = np.array(out_masks[0]).astype(bool)
                        masks[fidx] = m
                        if out_boxes is not None and len(out_boxes) > 0:
                            x, y, w, h = out_boxes[0]
                            boxes[fidx] = [x, y, x + w, y + h]
                        else:
                            H, W = self._frame_hw
                            ys, xs = np.where(m)
                            if len(ys) > 0:
                                boxes[fidx] = [xs.min()/W, ys.min()/H, xs.max()/W, ys.max()/H]

        finally:
            try:
                self.video_predictor.close_session(sid)
            except Exception:
                pass

        print(f"[PROP/{name}] tracked {len(masks)}/{len(session_frames)} frames")
        return masks, boxes

    # ──────────────────────────────────────────────────────
    # Absence 판정
    # ──────────────────────────────────────────────────────
    def _check_absence(self, name, mask, frame, area_drop_ratio=0.15, border_margin=5):
        history = self.mask_area_history[name]
        if len(history) < 2:
            return False, None
        prev_area = history[-2]
        curr_area = history[-1]
        if prev_area < 1.0 or (curr_area / (prev_area + 1e-8)) >= area_drop_ratio:
            return False, None
        if mask is None or curr_area == 0:
            return True, "vanished"
        H, W = frame.shape[:2]
        touches = (
            mask[:border_margin, :].any() or
            mask[-border_margin:, :].any() or
            mask[:, :border_margin].any() or
            mask[:, -border_margin:].any()
        )
        return (True, "out_of_frame") if touches else (True, "occluded")

    # ──────────────────────────────────────────────────────
    # 리셋
    # ──────────────────────────────────────────────────────
    def reset_session(self):
        self._recent_frames = []
        self.mask_area_history = {name: [] for name in self.object_names}
