"""
ObjectRegistry: 장면 내 객체의 상태를 추적·저장하는 자료구조.

ObjectState 필드:
  label            : 객체 이름 (str)
  presence         : 현재 프레임에 존재하는지 (float, 0 or 1)
  appearance       : CLIP 이미지 임베딩 (np.ndarray, shape (512,)) — rollback/redetect 전용
  bbox             : 마지막으로 본 bounding box [x1, y1, x2, y2] (np.ndarray, (4,))
  state            : 현재 상태 코드 (float)
                     0.0 = normal, 1.0 = interaction, -1.0 = absent
  last_good_frame  : 마지막 정상 프레임 (np.ndarray, (H, W, 3) uint8)
  last_good_mask   : 마지막 정상 마스크 (np.ndarray, (H, W) bool)
  shape_latent     : 현재 mask의 shape 특징 (np.ndarray, (SHAPE_SIZE^2,))
                     — bbox crop → SHAPE_SIZE×SHAPE_SIZE resize → flatten (위치 불변)
  shape_score      : 현재 mask와 last_good_shape 간 IoU (float, 0~1)
  last_good_shape  : last_good 시점의 shape_latent (np.ndarray)
  shape_rejected   : 이번 update에서 shape_score 미달로 last_good 갱신을 거부했는지 (bool)

to_feature_vector() → np.ndarray (FEATURE_DIM,):
  [presence(1), bbox_norm(4), state(1)] = 6-dim  (appearance 제외 — spatial 정보 중심)
  → ObjectStateEncoder MLP 입력으로 사용 (shape_latent는 별도 경로)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Optional

FEATURE_DIM  = 6    # presence(1) + bbox(4) + state(1)  — appearance 제외
MAX_OBJECTS  = 8    # 한 장면에서 최대 추적할 객체 수
SHAPE_SIZE    = 16    # shape_latent crop resize 크기 (32×32 = 1024-dim)
SHAPE_THRESH  = 0.65  # shape_score 이 값 미만이면 last_good 갱신 거부
AREA_THRESH   = 0.70  # area_ratio 이 값 미만이면 rollback anchor → initial
EXTENT_THRESH = 0.70  # extent_ratio 이 값 미만이면 rollback anchor → initial


@dataclass
class ObjectState:
    label: str
    presence: float = 0.0                          # 1.0 = 있음, 0.0 = 없음
    appearance: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    state: float = 0.0                             # 0=normal, 1=interaction, -1=absent
    last_good_frame: Optional[np.ndarray] = None   # (H, W, 3) uint8
    last_good_mask: Optional[np.ndarray] = None    # (H, W) bool
    shape_latent: Optional[np.ndarray] = None      # (SHAPE_SIZE^2,) 현재 shape
    shape_score: float = 1.0                       # 현재 vs last_good shape IoU
    last_good_shape: Optional[np.ndarray] = None   # last_good 시점 shape_latent
    shape_rejected: bool = False                   # 이번 update에서 last_good 갱신 거부됐는지
    initial_good_frame: Optional[np.ndarray] = None  # 첫 정상 프레임 이미지
    initial_good_mask:  Optional[np.ndarray] = None  # 첫 정상 프레임 mask
    initial_good_shape: Optional[np.ndarray] = None  # 첫 정상 프레임 shape_latent
    initial_good_bbox:  Optional[np.ndarray] = None  # 첫 정상 프레임 normalized bbox
    initial_area:   float = 0.0  # 첫 정상 프레임 mask pixel 수
    initial_extent: float = 0.0  # 첫 정상 프레임 bbox 픽셀 면적 (w*h)
    area_ratio:     float = 1.0  # current_area / initial_area
    extent_ratio:   float = 1.0  # current_extent / initial_extent
    # domain tracking (SR artifact가 downstream에 유입되지 않도록 감사용)
    detector_domain:    str   = 'original'  # 탐지 도메인: 'original', 'super_resolution', 'low_res'
    downstream_domain:  str   = 'original'  # 항상 'original' 유지
    scale_back_applied: bool  = False        # SR→original 좌표계 변환 적용 여부
    scale_factor:       float = 1.0          # enhance scale (1.0 = original 도메인)

    def to_feature_vector(self) -> np.ndarray:
        """(FEATURE_DIM=6,) float32 벡터로 직렬화. spatial 정보만 포함."""
        return np.concatenate([
            np.array([self.presence],  dtype=np.float32),   # (1,)
            self.bbox.astype(np.float32),                    # (4,)
            np.array([self.state],     dtype=np.float32),    # (1,)
        ])  # total: 6

    def to_shape_vector(self) -> np.ndarray:
        """(SHAPE_SIZE^2,) float32. shape_latent가 없으면 zeros."""
        if self.shape_latent is not None:
            return self.shape_latent.astype(np.float32)
        return np.zeros(SHAPE_SIZE * SHAPE_SIZE, dtype=np.float32)

    def update_good(self, frame: np.ndarray, mask: np.ndarray,
                    shape_latent: Optional[np.ndarray] = None,
                    area: float = None, extent: float = None):
        """정상 상태일 때 last_good_frame/mask/shape 갱신."""
        self.last_good_frame = frame.copy()
        self.last_good_mask  = mask.copy()
        if shape_latent is not None:
            self.last_good_shape = shape_latent.copy()
        self.presence = 1.0
        self.state    = 0.0
        # initial 값은 첫 호출 시 한 번만 세팅
        if self.initial_good_mask is None:
            self.initial_good_frame = frame.copy()
            self.initial_good_mask  = mask.copy()
            self.initial_good_shape = shape_latent.copy() if shape_latent is not None else None
            self.initial_good_bbox  = self.bbox.copy()
            self.initial_area   = area if area is not None else float(mask.sum())
            if extent is not None:
                self.initial_extent = extent
            else:
                ys, xs = np.where(mask)
                self.initial_extent = float(
                    (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
                ) if len(ys) > 0 else 0.0


class ObjectRegistry:
    """
    장면 내 여러 객체의 ObjectState를 관리하는 registry.

    Usage:
        registry = ObjectRegistry()
        registry.register("robot arm")
        registry.register("cup")
        registry.update("robot arm", presence=1.0, appearance=emb, bbox=box, state=0.0, frame=f, mask=m)
        tensor = registry.to_tensor()  # (N, FEATURE_DIM)
        padded = registry.to_padded_tensor(MAX_OBJECTS)  # (MAX_OBJECTS, FEATURE_DIM)
    """

    def __init__(self):
        self._objects: dict[str, ObjectState] = {}

    # ── 등록 / 조회 ─────────────────────────────────────────
    def register(self, label: str):
        if label not in self._objects:
            self._objects[label] = ObjectState(label=label)

    def get(self, label: str) -> ObjectState:
        return self._objects[label]

    def labels(self) -> list[str]:
        return list(self._objects.keys())

    def __len__(self):
        return len(self._objects)

    # ── 상태 갱신 ────────────────────────────────────────────
    def update(
        self,
        label: str,
        presence: float,
        appearance: np.ndarray,
        bbox: np.ndarray,
        state: float,
        frame: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        shape_latent: Optional[np.ndarray] = None,
        detector_domain:    str   = 'original',
        scale_back_applied: bool  = False,
        scale_factor:       float = 1.0,
    ):
        obj = self._objects[label]
        obj.presence   = presence
        obj.appearance = appearance.astype(np.float32)
        obj.bbox       = bbox.astype(np.float32)
        obj.state      = state
        obj.shape_rejected     = False
        obj.detector_domain    = detector_domain
        obj.downstream_domain  = 'original'   # 항상 original
        obj.scale_back_applied = scale_back_applied
        obj.scale_factor       = scale_factor

        # shape_score 계산
        if shape_latent is not None:
            obj.shape_latent = shape_latent
            if obj.last_good_shape is not None:
                obj.shape_score = ObjectRegistry.compute_shape_score(shape_latent, obj.last_good_shape)
            # last_good_shape 없으면 첫 프레임 — 무조건 신뢰
        else:
            obj.shape_score = 1.0

        # area / extent ratio 갱신
        curr_area = curr_extent = 0.0
        if mask is not None:
            curr_area = float(mask.sum())
            ys, xs = np.where(mask)
            if len(ys) > 0:
                curr_extent = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
            obj.area_ratio   = curr_area   / obj.initial_area   if obj.initial_area   > 0 else 1.0
            obj.extent_ratio = curr_extent / obj.initial_extent if obj.initial_extent > 0 else 1.0

        # absent나 interaction 아닌 정상 상태일 때 last_good 갱신
        # shape_score < SHAPE_THRESH 이면 갱신 거부 (robot arm은 예외)
        if frame is not None and mask is not None and presence > 0.5 and state >= 0:
            is_robot = "robot arm" in label.lower()
            if is_robot or obj.shape_score >= SHAPE_THRESH:
                obj.update_good(frame, mask, shape_latent, curr_area, curr_extent)
            else:
                obj.shape_rejected = True
                print(f"  [SHAPE] '{label}': shape_score={obj.shape_score:.3f} < {SHAPE_THRESH} → last_good 갱신 거부")

    def mark_absent(self, label: str):
        obj = self._objects[label]
        obj.presence = 0.0
        obj.state = -1.0

    # ── rollback ─────────────────────────────────────────────
    def rollback(self, label: str):
        """last_good_* 상태로 되돌린다."""
        obj = self._objects[label]
        if obj.last_good_frame is None:
            return
        obj.presence = 1.0
        obj.state = 0.0

    # ── snapshot / restore ───────────────────────────────────
    def snapshot(self):
        """전체 registry 상태를 deepcopy로 저장해 반환."""
        import copy
        return copy.deepcopy(self._objects)

    def restore(self, snap):
        """snapshot()으로 저장한 상태로 전체 복원."""
        import copy
        self._objects = copy.deepcopy(snap)

    # ── 텐서 변환 ────────────────────────────────────────────
    def to_tensor(self) -> np.ndarray:
        """(N_obj, FEATURE_DIM) float32."""
        if len(self._objects) == 0:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)
        return np.stack([obj.to_feature_vector() for obj in self._objects.values()])

    def to_padded_tensor(self, max_objects: int = MAX_OBJECTS) -> np.ndarray:
        """
        (max_objects, FEATURE_DIM) float32.
        객체 수가 max_objects보다 적으면 zeros로 패딩,
        많으면 앞에서 max_objects개만 사용.
        """
        feat = self.to_tensor()  # (N, FEATURE_DIM)
        N = feat.shape[0]
        padded = np.zeros((max_objects, FEATURE_DIM), dtype=np.float32)
        n_fill = min(N, max_objects)
        padded[:n_fill] = feat[:n_fill]
        return padded

    def to_padded_shape_tensor(self, max_objects: int = MAX_OBJECTS) -> np.ndarray:
        """
        (max_objects, SHAPE_SIZE^2) float32.
        객체 수가 max_objects보다 적으면 zeros로 패딩,
        많으면 앞에서 max_objects개만 사용.
        """
        dim = SHAPE_SIZE * SHAPE_SIZE
        if len(self._objects) == 0:
            return np.zeros((max_objects, dim), dtype=np.float32)
        vecs = [obj.to_shape_vector() for obj in self._objects.values()]
        N = len(vecs)
        padded = np.zeros((max_objects, dim), dtype=np.float32)
        n_fill = min(N, max_objects)
        padded[:n_fill] = np.stack(vecs[:n_fill])
        return padded

    # ── 유틸 ─────────────────────────────────────────────────
    def extract_appearance(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        clip_model,
        clip_processor,
        device: str = "cuda",
        context: str = "",
    ) -> np.ndarray:
        """
        mask로 crop한 객체 영역을 CLIP image encoder에 통과시켜
        (512,) appearance embedding을 반환한다.
        processor 입력은 항상 PIL RGB로 보장한다.
        """
        import torch
        from PIL import Image

        # ── mask가 없거나 비어 있으면 즉시 zeros 반환 ────────────
        if mask is None:
            print(f"[extract_appearance{context}] mask is None → zeros")
            return np.zeros(512, dtype=np.float32)

        ys, xs = np.where(mask)
        if len(ys) == 0:
            print(f"[extract_appearance{context}] empty mask → zeros")
            return np.zeros(512, dtype=np.float32)

        # ── bbox crop ───────────────────────────────────────────
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        # frame 자체의 형식 로그
        if isinstance(frame, np.ndarray):
            print(f"[extract_appearance{context}] frame shape={frame.shape} "
                  f"dtype={frame.dtype} min={frame.min()} max={frame.max()}")
        else:
            print(f"[extract_appearance{context}] frame type={type(frame)}")

        # invalid bbox: 1px 미만이면 skip
        if y2 <= y1 or x2 <= x1:
            print(f"[extract_appearance{context}] invalid bbox "
                  f"y=[{y1},{y2}] x=[{x1},{x2}] → zeros")
            return np.zeros(512, dtype=np.float32)

        crop = frame[y1:y2+1, x1:x2+1]
        if crop.size == 0:
            print(f"[extract_appearance{context}] empty crop → zeros")
            return np.zeros(512, dtype=np.float32)

        # ── PIL 변환 + RGB 강제 ──────────────────────────────────
        # crop이 2D(grayscale) / HxWx1 / HxWx4 등 비정상 채널일 경우 모두 RGB로 변환
        if isinstance(crop, np.ndarray):
            if crop.ndim == 2:                          # (H, W) grayscale
                crop = np.stack([crop, crop, crop], axis=-1)
            elif crop.ndim == 3 and crop.shape[2] == 1: # (H, W, 1)
                crop = np.repeat(crop, 3, axis=2)
            elif crop.ndim == 3 and crop.shape[2] == 4: # (H, W, 4) RGBA
                crop = crop[:, :, :3]
            if crop.dtype != np.uint8:
                crop = (crop * 255).clip(0, 255).astype(np.uint8) \
                       if crop.max() <= 1.0 else crop.clip(0, 255).astype(np.uint8)

        pil = Image.fromarray(crop)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")

        # processor 직전 최종 확인 로그
        print(f"[extract_appearance{context}] → processor: PIL mode={pil.mode} size={pil.size}")

        with torch.no_grad():
            inputs = clip_processor(images=pil, return_tensors="pt").to(device)
            output = clip_model(**inputs)
            emb = output.image_embeds[0].cpu().float().numpy().astype(np.float32)
        return emb

    @staticmethod
    def extract_shape_latent(mask: np.ndarray) -> np.ndarray:
        """
        binary mask → (SHAPE_SIZE^2,) float32, 위치 불변 shape 특징.
        bbox crop → SHAPE_SIZE×SHAPE_SIZE resize → flatten.
        빈 mask이면 zeros 반환.
        """
        from PIL import Image
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return np.zeros(SHAPE_SIZE * SHAPE_SIZE, dtype=np.float32)
        y1, y2, x1, x2 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        crop = mask[y1:y2+1, x1:x2+1].astype(np.uint8) * 255
        resized = np.array(
            Image.fromarray(crop).resize((SHAPE_SIZE, SHAPE_SIZE), Image.NEAREST)
        ).astype(np.float32) / 255.0
        return resized.flatten()

    @staticmethod
    def compute_shape_score(latent_a: np.ndarray, latent_b: np.ndarray) -> float:
        """두 shape_latent 간 IoU (0~1). 값이 낮을수록 shape이 크게 다름."""
        a = latent_a > 0.5
        b = latent_b > 0.5
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def mask_to_bbox(mask: np.ndarray, frame_hw: tuple[int, int]) -> np.ndarray:
        """
        (H, W) bool mask → 정규화된 bbox [x1, y1, x2, y2] ∈ [0, 1].
        빈 mask이면 zeros 반환.
        """
        H, W = frame_hw
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array([
            xs.min() / W, ys.min() / H,
            xs.max() / W, ys.max() / H,
        ], dtype=np.float32)
