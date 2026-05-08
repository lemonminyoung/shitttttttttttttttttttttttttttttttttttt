"""
ObjectStateEncoder — per-frame, per-object conditioning for Ctrl-World.

Pipeline per object slot:
  mask_crop (16×16=256) → ShapeProjector → 64
  concat [presence(1), bbox(4), state(1), shape_proj(64)] = 70
  → ObjectStateEncoder MLP → 1024

Output: (B, F, N, 1024)
  presence=0 slot → replaced by learned null_object token.

Not connected to UNet yet; used for dataloader/shape tests.
"""

import torch
import torch.nn as nn

N_OBJ   = 3    # fixed number of object slots
CROP_SZ = 16   # mask crop spatial size (CROP_SZ × CROP_SZ)
SHAPE_PROJ_DIM = 64
OBJ_FEAT_DIM   = 1 + 4 + 1 + SHAPE_PROJ_DIM   # presence + bbox + state + shape_proj = 70
HIDDEN_SIZE    = 1024


class ShapeProjector(nn.Module):
    """mask_crop flattened (256,) → (SHAPE_PROJ_DIM,)."""
    def __init__(self, crop_dim: int = CROP_SZ * CROP_SZ, proj_dim: int = SHAPE_PROJ_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(crop_dim, 128),
            nn.SiLU(),
            nn.Linear(128, proj_dim),
        )

    def forward(self, mask_crop_flat: torch.Tensor) -> torch.Tensor:
        """mask_crop_flat: (..., 256) → (..., 64)"""
        return self.proj(mask_crop_flat)


class ObjectStateEncoder(nn.Module):
    """
    Encodes per-frame, per-object state into 1024-dim tokens.

    Inputs (all tensors with leading dims B, F, N):
      presence:  (B, F, N, 1)     — 1.0=present, 0.0=absent
      bbox:      (B, F, N, 4)     — normalized [x1,y1,x2,y2]
      state:     (B, F, N, 1)     — 0=normal, 1=interaction, -1=absent
      mask_crop: (B, F, N, 16,16) — binary mask crop

    Output: (B, F, N, 1024)
      Absent slots (presence==0) → null_object token (learned).
    """

    def __init__(self,
                 obj_feat_dim: int = OBJ_FEAT_DIM,
                 hidden_size:  int = HIDDEN_SIZE,
                 n_objects:    int = N_OBJ,
                 crop_sz:      int = CROP_SZ):
        super().__init__()
        self.n_objects = n_objects
        self.crop_sz   = crop_sz

        self.shape_projector = ShapeProjector(
            crop_dim=crop_sz * crop_sz,
            proj_dim=SHAPE_PROJ_DIM,
        )

        self.encoder = nn.Sequential(
            nn.Linear(obj_feat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_size),
        )
        # Default PyTorch init (kaiming_uniform) — presence=1 inputs must produce non-zero tokens.
        # Zero-init belongs only on adapter.out_proj, not here.
        # nn.init.zeros_(self.encoder[-1].weight)
        # nn.init.zeros_(self.encoder[-1].bias)

        # Learned null token — small normal so absent slots start non-zero
        # self.null_object = nn.Parameter(torch.zeros(hidden_size))
        self.null_object = nn.Parameter(torch.empty(hidden_size).normal_(0, 0.02))

    def forward(
        self,
        presence:  torch.Tensor,   # (B, F, N, 1)
        bbox:      torch.Tensor,   # (B, F, N, 4)
        state:     torch.Tensor,   # (B, F, N, 1)
        mask_crop: torch.Tensor,   # (B, F, N, 16, 16)
    ) -> torch.Tensor:             # (B, F, N, 1024)

        B, F, N = presence.shape[:3]

        # Flatten mask crop: (B, F, N, 256)
        mask_flat = mask_crop.flatten(-2)                      # (B, F, N, 256)
        shape_proj = self.shape_projector(mask_flat)           # (B, F, N, 64)

        # Build per-slot feature vector: [presence, bbox, state, shape_proj]
        feat = torch.cat([presence, bbox, state, shape_proj], dim=-1)  # (B, F, N, 70)

        encoded = self.encoder(feat)                           # (B, F, N, 1024)

        # Replace absent slots with null_object token
        is_present = (presence > 0.5)                         # (B, F, N, 1) bool
        null = self.null_object.view(1, 1, 1, -1).expand(B, F, N, -1)
        out = torch.where(is_present, encoded, null)          # (B, F, N, 1024)

        return out
