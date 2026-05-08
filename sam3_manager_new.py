# sam3_manager_new.py
"""
SAM3 멀티오브젝트 tracker — 정석 버전.

설계 원칙:
  - SAM3 내부 패치 없음. 공개 API만 사용.
  - 객체마다 완전히 독립된 세션. 세션 간 상태 공유 없음.
  - initialize(): 객체별 독립 세션 + text prompt → mask/box 탐지
  - update_chunk(): 객체별 독립 세션 + anchor box prompt → propagate
  - redetect(): 특정 객체를 현재 프레임에서 text prompt로 재탐지

슬라이딩 윈도우:
  - [anchor(객체별 기준 프레임)] + [최근 window_size 프레임] + [새 chunk]
  - redetect() 성공 시 해당 객체의 anchor frame/box가 새 위치로 갱신됨
"""

import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image


class SAM3ManagerNew:

    def __init__(self, checkpoint_path: str, device: str = "cuda", window_size: int = 4):
        from sam3.model_builder import build_sam3_video_predictor

        # device에서 GPU 번호 추출
        if device.startswith("cuda:"):
            gpu_id = int(device.split(":")[1])
            gpus = [gpu_id]
        else:
            gpus = None  # 기본값 (cuda:0)

        self.predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path, gpus_to_use=gpus)
        self.window_size = window_size

        # 객체 목록
        self.object_names: list = []

        # 객체별 상태
        self.object_masks: dict  = {}   # {name: (H,W) bool | None}
        self.mask_area_history: dict = {}  # {name: [float, ...]}
        self.initial_areas: dict = {}   # {name: float} — initialize()에서 고정
        self._anchor_boxes: dict = {}   # {name: [x1,y1,x2,y2] | None}
        self._anchor_frames: dict = {}  # {name: np.ndarray} — 객체별 기준 프레임 (재탐지 시 갱신)
        self.last_good_boxes: dict = {} # {name: [x1,y1,x2,y2] | None}
        self._centroid_history: dict = {}  # {name: [(cx,cy)|None, ...]} 최근 10개

        # 슬라이딩 윈도우
        self._recent_frames: list = []
        self._frame_hw = None
        self._tmpdir = None

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def initialize(self, frame: np.ndarray, object_names: list,
                   points_dict: dict = None):
        """실제 첫 프레임에서 각 객체를 탐지, anchor box 고정.

        points_dict: {name: {"point_coords": [[x,y],...], "point_labels": [1,0,...]}}
          - 해당 객체는 text 탐지 후 동일 세션 내에서 point로 mask refinement 추가 수행
          - None이거나 name 없으면 text prompt만 사용
        """
        self.object_names = object_names
        self.object_masks = {}
        self.mask_area_history = {n: [] for n in object_names}
        self.initial_areas = {}
        self._anchor_boxes = {}
        self._anchor_frames = {}
        self.last_good_boxes = {}
        self._centroid_history = {}
        self._recent_frames = []
        self._frame_hw = frame.shape[:2]

        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = tempfile.mkdtemp()
        Image.fromarray(frame).save(os.path.join(self._tmpdir, "00000.jpg"))

        for name in object_names:
            pt_info = (points_dict or {}).get(name)
            mask, box = self._detect_one(
                name, self._tmpdir,
                point_coords=pt_info["point_coords"] if pt_info else None,
                point_labels=pt_info["point_labels"] if pt_info else None,
            )
            self.object_masks[name] = mask
            self._anchor_boxes[name] = box
            self._anchor_frames[name] = frame          # 초기 anchor frame
            self.last_good_boxes[name] = list(box) if box else None
            area = float(mask.sum()) if mask is not None else 0.0
            self.mask_area_history[name].append(area)
            self.initial_areas[name] = area
            if mask is not None:
                ys, xs = np.where(mask)
                centroid = (float(xs.mean()), float(ys.mean())) if len(ys) > 0 else None
            else:
                centroid = None
            self._centroid_history[name] = [centroid] if centroid is not None else []
            print(f"[INIT] '{name}': area={area:.0f}  box={box}")

    def update_chunk(self, frames: list) -> list:
        """
        생성된 프레임 chunk를 받아 객체별로 추적.
        객체마다 독립적인 anchor frame을 사용 (재탐지 후 anchor가 달라질 수 있음).
        returns: list of {name: {"mask", "area", "absent", "cause", "first_bad_t"}}
        """
        assert self.object_names, "initialize() 먼저 호출 필요"

        recent = self._recent_frames[-self.window_size:] if self.window_size > 0 else []
        n_anchor = 1 + len(recent)   # 새 chunk가 시작되는 frame index

        # 객체별 독립 propagation — 각자 anchor frame이 다를 수 있으므로 per-object tmpdir
        all_masks = {}   # {fidx: {name: mask}}
        all_boxes = {}   # {name: {fidx: box}}
        for name in self.object_names:
            anchor_frame = self._anchor_frames.get(name)
            if anchor_frame is None:
                print(f"[WARN] '{name}': anchor_frame 없음, skip")
                continue
            session_frames = [anchor_frame] + recent + frames
            print(f"[SAM3] '{name}': anchor(1) + recent({len(recent)}) + new({len(frames)}) = {len(session_frames)}")

            obj_tmpdir = tempfile.mkdtemp()
            try:
                for i, f in enumerate(session_frames):
                    Image.fromarray(f).save(os.path.join(obj_tmpdir, f"{i:05d}.jpg"))
                masks, boxes = self._propagate_one(name, len(session_frames), obj_tmpdir)
            finally:
                shutil.rmtree(obj_tmpdir, ignore_errors=True)

            all_boxes[name] = boxes
            for fidx, m in masks.items():
                all_masks.setdefault(fidx, {})[name] = m

        # last_good_boxes 갱신
        for name in self.object_names:
            bdict = all_boxes.get(name, {})
            if bdict:
                self.last_good_boxes[name] = bdict[max(bdict)]

        # chunk 첫 프레임의 area를 baseline으로 교체
        for name in self.object_names:
            m = all_masks.get(n_anchor, {}).get(name)
            area = float(m.sum()) if m is not None else 0.0
            if self.mask_area_history[name]:
                self.mask_area_history[name][-1] = area

        # 프레임별 결과 조립
        results_list = []
        first_bad_t = None
        for t, frame in enumerate(frames):
            fidx = n_anchor + t
            frame_results = {}
            for name in self.object_names:
                mask = all_masks.get(fidx, {}).get(name)
                area = float(mask.sum()) if mask is not None else 0.0
                self.mask_area_history[name].append(area)
                self.object_masks[name] = mask

                # centroid 기록 (최근 10개 유지)
                if mask is not None:
                    ys_c, xs_c = np.where(mask)
                    centroid = (float(xs_c.mean()), float(ys_c.mean())) if len(ys_c) > 0 else None
                else:
                    centroid = None
                hist = self._centroid_history.setdefault(name, [])
                hist.append(centroid)
                if len(hist) > 10:
                    hist.pop(0)

                other_masks = {n: all_masks.get(fidx, {}).get(n)
                               for n in self.object_names if n != name}
                absent, cause = self._check_absence(name, mask, frame, other_masks=other_masks)
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

        self._recent_frames = (self._recent_frames + frames)[-self.window_size:]
        return results_list

    def redetect(self, frame: np.ndarray, name: str,
                 roi_pad_ratio: float = 1.5, roi_min_pad: int = 40) -> tuple:
        """
        현재 프레임에서 객체 재탐지.

        전략:
          1) last_good_boxes 기준으로 padded ROI crop → SAM3 탐지
             - roi_pad_ratio: bbox 크기의 몇 배를 패딩으로 사용 (기본 1.5×)
             - roi_min_pad:   최소 패딩 픽셀 (기본 40px)
          2) ROI 탐지 실패 시 전체 프레임 fallback

        탐지 성공 시 anchor frame/box 갱신.
        returns: (mask, box) | (None, None)  — 좌표는 전체 프레임 기준
        """
        H, W = frame.shape[:2]
        self._frame_hw = (H, W)

        # ── 1단계: ROI 기반 탐지 ──────────────────────────────────────────────
        last_box = self.last_good_boxes.get(name)   # normalized [x1,y1,x2,y2]
        if last_box is not None:
            x1n, y1n, x2n, y2n = last_box
            x1p = int(x1n * W); y1p = int(y1n * H)
            x2p = int(x2n * W); y2p = int(y2n * H)

            bw = max(1, x2p - x1p)
            bh = max(1, y2p - y1p)

            # 패딩: bbox 크기의 roi_pad_ratio 배, 최소 roi_min_pad px
            pad_x = max(roi_min_pad, int(bw * roi_pad_ratio))
            pad_y = max(roi_min_pad, int(bh * roi_pad_ratio))

            rx1 = max(0, x1p - pad_x)
            ry1 = max(0, y1p - pad_y)
            rx2 = min(W, x2p + pad_x)
            ry2 = min(H, y2p + pad_y)

            crop = frame[ry1:ry2, rx1:rx2]
            cH, cW = crop.shape[:2]
            print(f"[REDETECT/ROI] '{name}': bbox_px=({x1p},{y1p},{x2p},{y2p})"
                  f"  pad=({pad_x},{pad_y})  roi=({rx1},{ry1},{rx2},{ry2})")

            self._frame_hw = (cH, cW)   # _detect_one이 crop 해상도로 좌표 계산
            tmpdir = tempfile.mkdtemp()
            try:
                Image.fromarray(crop).save(os.path.join(tmpdir, "00000.jpg"))
                mask_crop, box_crop = self._detect_one(name, tmpdir)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
            self._frame_hw = (H, W)     # 복원

            if mask_crop is not None and box_crop is not None:
                # mask: crop 좌표 → 전체 프레임 좌표
                mask_full = np.zeros((H, W), dtype=bool)
                mask_full[ry1:ry2, rx1:rx2] = mask_crop

                # box: normalized crop 좌표 → normalized 전체 프레임 좌표
                bx1n, by1n, bx2n, by2n = box_crop
                box_full = [
                    (rx1 + bx1n * cW) / W,
                    (ry1 + by1n * cH) / H,
                    (rx1 + bx2n * cW) / W,
                    (ry1 + by2n * cH) / H,
                ]
                self._commit_redetect(name, frame, mask_full, box_full)
                area = float(mask_full.sum())
                print(f"[REDETECT/ROI] '{name}': found  area={area:.0f}  box={[round(v,3) for v in box_full]}")
                return mask_full, box_full
            else:
                print(f"[REDETECT/ROI] '{name}': not found in ROI → fallback to full frame")

        # ── 2단계: 전체 프레임 fallback ──────────────────────────────────────
        tmpdir = tempfile.mkdtemp()
        try:
            Image.fromarray(frame).save(os.path.join(tmpdir, "00000.jpg"))
            mask, box = self._detect_one(name, tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if mask is not None and box is not None:
            self._commit_redetect(name, frame, mask, box)
            print(f"[REDETECT/full] '{name}': found  area={float(mask.sum()):.0f}  box={box}")
        else:
            print(f"[REDETECT] '{name}': not found")

        return mask, box

    def _commit_redetect(self, name: str, frame: np.ndarray,
                         mask: np.ndarray, box: list):
        """재탐지 성공 시 내부 상태 갱신 (공통 처리)."""
        self._anchor_frames[name] = frame
        self._anchor_boxes[name]  = box
        self.object_masks[name]   = mask
        area = float(mask.sum())
        if self.mask_area_history.get(name):
            self.mask_area_history[name][-1] = area

    def set_anchor(self, name: str, frame: np.ndarray, box: list):
        """
        SAM 탐지 없이 anchor frame/box를 직접 지정.
        rollback 후 last_good 기준으로 tracking 재개할 때 사용.
        """
        self._anchor_frames[name] = frame
        self._anchor_boxes[name] = box
        print(f"[ANCHOR] '{name}': restored  box={[round(v, 4) for v in box]}")

    def reset_session(self):
        self._recent_frames = []
        self.mask_area_history = {n: [] for n in self.object_names}
        self._centroid_history = {}

    # ────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────

    def _detect_one(self, name: str, video_dir: str,
                    point_coords=None, point_labels=None):
        """독립 세션에서 객체 하나 탐지. (mask, box) 반환.

        point_coords가 있으면 point-only로 새 객체 탐지 (SAM3 공식 예제 방식).
        없으면 text prompt 탐지.

        SAM3 공식 예제 (sam3.1_video_predictor_example.ipynb cell-25~26):
          points_tensor = torch.tensor(
              [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in points_abs],
              dtype=torch.float32,
          )
          predictor.add_prompt(session_id, frame_index=0,
                               points=points_tensor, point_labels=..., obj_id=<new_id>)
        """
        H, W = self._frame_hw
        sid = self.predictor.start_session(video_dir)["session_id"]
        mask, box = None, None
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if point_coords is not None:
                    # ── point → bbox 변환 ────────────────────────────────
                    # SAM3의 add_prompt(points=..., obj_id=...) 경로는
                    # add_tracker_new_points → _build_tracker_output을 타는데,
                    # 이 경로는 propagate_in_video 캐시가 선행되어야 함.
                    # _detect_one은 독립 세션이므로 캐시가 없어 항상 AssertionError.
                    # bbox prompt 경로는 캐시 불필요 → point_coords에서 bbox 유도.
                    xs_px = [p[0] for p in point_coords]
                    ys_px = [p[1] for p in point_coords]
                    x1_px = max(0,   min(xs_px))
                    y1_px = max(0,   min(ys_px))
                    x2_px = min(W,   max(xs_px))
                    y2_px = min(H,   max(ys_px))
                    # 10% padding (최소 5px)
                    pw = max(5, (x2_px - x1_px) * 0.10)
                    ph = max(5, (y2_px - y1_px) * 0.10)
                    x1_px = max(0, x1_px - pw)
                    y1_px = max(0, y1_px - ph)
                    x2_px = min(W, x2_px + pw)
                    y2_px = min(H, y2_px + ph)
                    bx = x1_px / W
                    by = y1_px / H
                    bw = (x2_px - x1_px) / W
                    bh = (y2_px - y1_px) / H
                    print(f"[DETECT/point→bbox] '{name}': "
                          f"points bbox px=({int(x1_px)},{int(y1_px)})-({int(x2_px)},{int(y2_px)}) "
                          f"norm=({bx:.3f},{by:.3f},{bw:.3f},{bh:.3f})")
                    r = self.predictor.add_prompt(
                        sid, frame_idx=0,
                        bounding_boxes=[[bx, by, bw, bh]],
                        bounding_box_labels=[1],
                    )
                    mode = "point→bbox"
                else:
                    # ── text-only ────────────────────────────────────────
                    r = self.predictor.add_prompt(sid, frame_idx=0, text=name)
                    mode = "text"

            out   = r["outputs"]
            masks = out.get("out_binary_masks", [])
            boxes = out.get("out_boxes_xywh")
            obj_ids = list(out.get("out_obj_ids", []))
            print(f"[DETECT/{mode}] '{name}': obj_ids={obj_ids}  masks={len(masks)}")

            if len(masks) > 0:
                mask = np.array(masks[-1]).astype(bool)
                if boxes is not None and len(boxes) > 0:
                    x, y, w, h = boxes[-1]
                    box = [x, y, x + w, y + h]
                else:
                    ys, xs = np.where(mask)
                    if len(ys) > 0:
                        box = [xs.min()/W, ys.min()/H, xs.max()/W, ys.max()/H]
        finally:
            try:
                self.predictor.close_session(sid)
            except Exception:
                pass
        return mask, box

    def _propagate_one(self, name: str, n_frames: int, tmpdir: str):
        """독립 세션에서 anchor box prompt → forward propagate. (masks, boxes) 반환."""
        H, W = self._frame_hw
        box = self._anchor_boxes.get(name)
        sid = self.predictor.start_session(tmpdir)["session_id"]
        masks, boxes = {}, {}
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if box is not None:
                    x1, y1, x2, y2 = box
                    r = self.predictor.add_prompt(
                        sid, frame_idx=0,
                        bounding_boxes=[[x1, y1, x2 - x1, y2 - y1]],
                        bounding_box_labels=[1],
                    )
                else:
                    r = self.predictor.add_prompt(sid, frame_idx=0, text=name)
            out = r["outputs"]
            print(f"[TRACK] '{name}': obj_ids={list(out.get('out_obj_ids',[]))}  box={box}")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for result in self.predictor.propagate_in_video(sid, propagation_direction="forward"):
                    fidx = result["frame_index"]
                    o = result["outputs"]
                    obj_ids = o.get("out_obj_ids", [])
                    out_masks = o.get("out_binary_masks", [])
                    out_boxes = o.get("out_boxes_xywh")
                    if len(obj_ids) > 0 and len(out_masks) > 0:
                        m = np.array(out_masks[0]).astype(bool)
                        masks[fidx] = m
                        if out_boxes is not None and len(out_boxes) > 0:
                            x, y, w, h = out_boxes[0]
                            boxes[fidx] = [x, y, x + w, y + h]
                        else:
                            ys, xs = np.where(m)
                            if len(ys) > 0:
                                boxes[fidx] = [xs.min()/W, ys.min()/H, xs.max()/W, ys.max()/H]
        finally:
            try:
                self.predictor.close_session(sid)
            except Exception:
                pass
        print(f"[TRACK] '{name}': tracked {len(masks)}/{n_frames} frames")
        return masks, boxes

    def _check_absence(self, name: str, mask, frame: np.ndarray,
                       other_masks: dict = None,
                       area_drop_ratio: float = 0.15, area_surge_ratio: float = 1.7,
                       border_margin: int = 5):
        history = self.mask_area_history[name]
        if len(history) < 2:
            return False, None
        prev, curr = history[-2], history[-1]
        # mask 자체가 없으면 무조건 vanished — area 비교 전에 먼저 체크
        if mask is None:
            return True, "vanished"
        # Area surge (1): 직전 프레임 대비 1.7배 이상 — 급격한 증가
        if prev > 10.0 and curr >= prev * area_surge_ratio:
            return True, "crushed"
        # Area surge (2): 초기 면적 대비 1.7배 이상 — 점진적 drift도 포착
        initial = self.initial_areas.get(name, 0.0)
        if initial > 10.0 and curr >= initial * area_surge_ratio:
            return True, "crushed"
        if prev < 1.0 or (curr / (prev + 1e-8)) >= area_drop_ratio:
            return False, None
        if curr == 0:
            return True, "vanished"

        # ── 다른 object mask가 줄어든 영역을 채우고 있는지 검사 ──────
        # 현재 mask와 다른 object mask의 union이 30% 이상 겹치면 occluded
        if other_masks:
            other_union = None
            for om in other_masks.values():
                if om is not None:
                    other_union = om if other_union is None else (other_union | om)
            if other_union is not None:
                covered = int((mask & other_union).sum())
                if curr > 0 and covered / curr > 0.3:
                    print(f"  [ABSENCE] '{name}': area drop but {covered}/{int(curr)} px covered by other → occluded")
                    return True, "occluded"

        # ── border 접촉 + 이동 방향으로 out_of_frame 판정 ─────────────
        H, W = frame.shape[:2]
        touches_top    = bool(mask[:border_margin, :].any())
        touches_bottom = bool(mask[-border_margin:, :].any())
        touches_left   = bool(mask[:, :border_margin].any())
        touches_right  = bool(mask[:, -border_margin:].any())
        touches = touches_top or touches_bottom or touches_left or touches_right

        if touches:
            vel = self._centroid_velocity(name)
            if vel is not None:
                vx, vy = vel
                # 속도가 border 방향을 향할 때만 out_of_frame
                heading_out = (
                    (touches_top    and vy < -0.5) or
                    (touches_bottom and vy >  0.5) or
                    (touches_left   and vx < -0.5) or
                    (touches_right  and vx >  0.5)
                )
                if not heading_out:
                    print(f"  [ABSENCE] '{name}': border touch but vel=({vx:.1f},{vy:.1f}) not heading out → occluded")
                    return True, "occluded"
            return True, "out_of_frame"

        return True, "occluded"

    def _centroid_velocity(self, name: str, n: int = 4):
        """최근 n 구간 유효 centroid로 평균 이동 속도 (vx, vy) 반환. 불가면 None."""
        valid = [p for p in self._centroid_history.get(name, []) if p is not None]
        if len(valid) < 2:
            return None
        recent = valid[-(n + 1):]
        vx = float(np.mean([recent[i+1][0] - recent[i][0] for i in range(len(recent)-1)]))
        vy = float(np.mean([recent[i+1][1] - recent[i][1] for i in range(len(recent)-1)]))
        return vx, vy
