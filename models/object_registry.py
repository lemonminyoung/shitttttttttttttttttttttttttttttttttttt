"""
ObjectRegistry: 장면 내 객체의 상태를 추적·저장하는 자료구조.

ObjectState 필드:
  label           : 객체 이름 (str)
  presence        : 현재 프레임에 존재하는지 (float, 0 or 1)
  appearance      : CLIP 이미지 임베딩 (np.ndarray, shape (512,))
                    — SAM3 mask crop → CLIP image encoder로 추출
  bbox            : 마지막으로 본 bounding box [x1, y1, x2, y2] (np.ndarray, (4,))
                    — 위치/크기 정보를 shape descriptor로 활용
  state           : 현재 상태 코드 (float)
                    0.0 = normal, 1.0 = interaction, -1.0 = absent
  last_good_frame : 마지막 정상 프레임 (np.ndarray, (H, W, 3) uint8)
  last_good_mask  : 마지막 정상 마스크 (np.ndarray, (H, W) bool)

to_feature_vector() → np.ndarray (FEATURE_DIM,):
  [presence(1), appearance(512), bbox_norm(4), state(1)] = 518-dim
  → ObjectStateEncoder MLP 입력으로 사용
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Optional

FEATURE_DIM = 518   # presence(1) + appearance(512) + bbox(4) + state(1)
MAX_OBJECTS = 8     # 한 장면에서 최대 추적할 객체 수


@dataclass
class ObjectState:
    label: str
    presence: float = 0.0                          # 1.0 = 있음, 0.0 = 없음
    appearance: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    state: float = 0.0                             # 0=normal, 1=interaction, -1=absent
    last_good_frame: Optional[np.ndarray] = None   # (H, W, 3) uint8
    last_good_mask: Optional[np.ndarray] = None    # (H, W) bool

    def to_feature_vector(self) -> np.ndarray:
        """(FEATURE_DIM,) float32 벡터로 직렬화."""
        return np.concatenate([
            np.array([self.presence],  dtype=np.float32),   # (1,)
            self.appearance.astype(np.float32),              # (512,)
            self.bbox.astype(np.float32),                    # (4,)
            np.array([self.state],     dtype=np.float32),    # (1,)
        ])  # total: 518

    def update_good(self, frame: np.ndarray, mask: np.ndarray):
        """정상 상태일 때 last_good_frame/mask 갱신."""
        self.last_good_frame = frame.copy()
        self.last_good_mask = mask.copy()
        self.presence = 1.0
        self.state = 0.0


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
    ):
        obj = self._objects[label]
        obj.presence = presence
        obj.appearance = appearance.astype(np.float32)
        obj.bbox = bbox.astype(np.float32)
        obj.state = state
        if frame is not None and mask is not None and presence > 0.5 and state >= 0:
            obj.update_good(frame, mask)

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

    # ── 유틸 ─────────────────────────────────────────────────
    def extract_appearance(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        clip_model,
        clip_processor,
        device: str = "cuda",
    ) -> np.ndarray:
        """
        mask로 crop한 객체 영역을 CLIP image encoder에 통과시켜
        (512,) appearance embedding을 반환한다.
        """
        import torch
        from PIL import Image

        # mask crop: bbox 기준으로 잘라냄
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return np.zeros(512, dtype=np.float32)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = frame[y1:y2+1, x1:x2+1]
        pil = Image.fromarray(crop)

        with torch.no_grad():
            inputs = clip_processor(images=pil, return_tensors="pt").to(device)
            # CLIPVisionModelWithProjection: output.image_embeds (1, 512)
            output = clip_model(**inputs)
            emb = output.image_embeds[0].cpu().numpy().astype(np.float32)
        return emb

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
