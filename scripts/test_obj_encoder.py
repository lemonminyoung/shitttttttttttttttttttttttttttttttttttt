"""
Shape + correctness test for ObjectStateEncoder and dataloader object fields.

Checks:
  1. Mock batch field shapes are all correct
  2. ObjectStateEncoder output shape = [B, 7, 3, 1024]
  3. presence=0 slot token == null_object (learned token)
  4. presence=1 dummy: bbox/mask gradients flow (normal init override)
  5. Real Dataset_mix batch: key existence + shapes + statistics + NaN/Inf
  6. ObjectStateEncoder with real batch → shape (B, F_CTX, N_OBJ, 1024)
  7. _encode_objects with real batch → shape (B, num_history+num_frames, N_OBJ, 1024)
  8. null_object gradient check via absent slot in loss

Usage:
  cd /home/dgu/minyoung/Ctrl-World
  python scripts/test_obj_encoder.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.object_state_encoder import ObjectStateEncoder, N_OBJ, CROP_SZ

NUM_HISTORY = 6
NUM_FRAMES  = 5
F_CTX       = NUM_HISTORY + 1   # 7 context frames

B = 4
all_pass = True


def check(name, cond, extra=""):
    global all_pass
    status = "PASS" if cond else "FAIL"
    if not cond:
        all_pass = False
    print(f"  {status}  {name}" + (f"  ({extra})" if extra else ""))


# ── 1. Mock batch field shapes ────────────────────────────────────────────────
print("=" * 60)
print("[1] Mock batch field shapes")
print("=" * 60)

mock_batch = {
    "object_presence":  torch.zeros(B, F_CTX, N_OBJ, 1),
    "object_bbox":      torch.zeros(B, F_CTX, N_OBJ, 4),
    "object_state":     torch.zeros(B, F_CTX, N_OBJ, 1),
    "object_mask_crop": torch.zeros(B, F_CTX, N_OBJ, CROP_SZ, CROP_SZ),
}

expected_shapes = {
    "object_presence":  (B, F_CTX, N_OBJ, 1),
    "object_bbox":      (B, F_CTX, N_OBJ, 4),
    "object_state":     (B, F_CTX, N_OBJ, 1),
    "object_mask_crop": (B, F_CTX, N_OBJ, CROP_SZ, CROP_SZ),
}

for key, exp in expected_shapes.items():
    got = tuple(mock_batch[key].shape)
    check(f"{key}: {exp}", got == exp, f"got {got}")


# ── 2. ObjectStateEncoder output shape = [B, 7, 3, 1024] ─────────────────────
print()
print("=" * 60)
print("[2] ObjectStateEncoder output shape = [B, 7, 3, 1024]")
print("=" * 60)

encoder = ObjectStateEncoder()
encoder.eval()

presence  = mock_batch["object_presence"]
bbox      = mock_batch["object_bbox"]
state     = mock_batch["object_state"]
mask_crop = mock_batch["object_mask_crop"]

with torch.no_grad():
    out = encoder(presence, bbox, state, mask_crop)

exp_shape = (B, F_CTX, N_OBJ, 1024)
check("output shape", tuple(out.shape) == exp_shape,
      f"expected {exp_shape}, got {tuple(out.shape)}")


# ── 3. presence=0 → token == null_object ─────────────────────────────────────
print()
print("=" * 60)
print("[3] presence=0 slot == null_object token")
print("=" * 60)

with torch.no_grad():
    out_absent = encoder(presence, bbox, state, mask_crop)

null_val = encoder.null_object.detach()
null_bc  = null_val.view(1, 1, 1, -1).expand_as(out_absent)
is_null  = torch.allclose(out_absent, null_bc)
check("all-absent output == null_object", is_null)


# ── 4. presence=1 dummy: bbox/mask gradients flow ────────────────────────────
print()
print("=" * 60)
print("[4] presence=1 dummy — bbox & mask gradients flow")
print("=" * 60)

encoder.train()

presence_g  = presence.clone()
presence_g[:, :, 0, :] = 1.0          # slot 0 present

bbox_g      = bbox.clone().requires_grad_(True)
mask_crop_g = mask_crop.clone().requires_grad_(True)
state_g     = state.clone()

out_g = encoder(presence_g, bbox_g, state_g, mask_crop_g)
loss  = out_g[:, :, 0, :].sum()       # slot 0 (present) → encoder path
loss.backward()

check("bbox gradient flows through present-slot encoder path",
      bbox_g.grad is not None and bbox_g.grad.abs().sum() > 0)
check("mask_crop gradient flows through shape_projector → encoder path",
      mask_crop_g.grad is not None and mask_crop_g.grad.abs().sum() > 0)
check("null_object receives gradient (absent slot 1, via torch.where)",
      encoder.null_object.grad is not None)


# ── 5. Real Dataset_mix batch ─────────────────────────────────────────────────
print()
print("=" * 60)
print("[5] Real Dataset_mix batch — keys / shapes / statistics / NaN-Inf")
print("=" * 60)

import warnings
warnings.filterwarnings("ignore")

from config import wm_args
from dataset.dataset_droid_exp33 import Dataset_mix

args = wm_args()
ds = Dataset_mix(args, mode="train")
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
real_batch = next(iter(loader))

REQUIRED_KEYS = [
    "latent", "action", "text",
    "object_presence", "object_bbox", "object_state", "object_mask_crop",
]
for k in REQUIRED_KEYS:
    check(f"key '{k}' exists", k in real_batch)

# shape checks
B_real = 2
exp_real = {
    "latent":           (B_real, NUM_HISTORY + NUM_FRAMES, 4, 72, 40),
    "action":           (B_real, NUM_HISTORY + NUM_FRAMES, 7),
    "object_presence":  (B_real, F_CTX, N_OBJ, 1),
    "object_bbox":      (B_real, F_CTX, N_OBJ, 4),
    "object_state":     (B_real, F_CTX, N_OBJ, 1),
    "object_mask_crop": (B_real, F_CTX, N_OBJ, CROP_SZ, CROP_SZ),
}
for k, exp in exp_real.items():
    if k not in real_batch:
        continue
    got = tuple(real_batch[k].shape)
    check(f"{k} shape {exp}", got == exp, f"got {got}")

# statistics
p  = real_batch["object_presence"]
bb = real_batch["object_bbox"]
mk = real_batch["object_mask_crop"]

print(f"  INFO  object_presence  mean={p.mean():.4f}  "
      f"(non-zero frames={int((p > 0.5).sum())})")
print(f"  INFO  object_mask_crop sum={mk.sum():.1f}  "
      f"(non-zero elements={int((mk > 0).sum())})")
print(f"  INFO  object_bbox      min={bb.min():.4f}  max={bb.max():.4f}")

# NaN / Inf
for k in ["latent", "action", "object_presence", "object_bbox", "object_mask_crop"]:
    if k not in real_batch:
        continue
    t = real_batch[k]
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    check(f"no NaN/Inf in {k}", not has_nan and not has_inf,
          f"nan={has_nan} inf={has_inf}")


# ── 6. ObjectStateEncoder with real batch → (B, F_CTX, N_OBJ, 1024) ─────────
print()
print("=" * 60)
print("[6] ObjectStateEncoder — real batch → (B, F_CTX, N_OBJ, 1024)")
print("=" * 60)

encoder.eval()
with torch.no_grad():
    real_out = encoder(
        real_batch["object_presence"],
        real_batch["object_bbox"],
        real_batch["object_state"],
        real_batch["object_mask_crop"],
    )

exp6 = (B_real, F_CTX, N_OBJ, 1024)
check(f"output shape {exp6}", tuple(real_out.shape) == exp6,
      f"got {tuple(real_out.shape)}")
check("no NaN/Inf in encoder output",
      not torch.isnan(real_out).any() and not torch.isinf(real_out).any())


# ── 7. _encode_objects → (B, num_history+num_frames, N_OBJ, 1024) ───────────
print()
print("=" * 60)
print("[7] _encode_objects — future repeat → (B, H+F, N_OBJ, 1024)")
print("=" * 60)

from scripts.train_phase1 import _encode_objects

encoder.eval()
with torch.no_grad():
    ohs = _encode_objects(
        encoder, real_batch,
        num_history=NUM_HISTORY, num_future=NUM_FRAMES,
        device="cpu", dtype=torch.float32,
    )

exp7 = (B_real, NUM_HISTORY + NUM_FRAMES, N_OBJ, 1024)
check(f"output shape {exp7}", tuple(ohs.shape) == exp7,
      f"got {tuple(ohs.shape)}")

# context and future slices
ctx_slice    = ohs[:, :F_CTX, :, :]
future_slice = ohs[:, F_CTX:, :, :]
current_obj  = ohs[:, NUM_HISTORY:NUM_HISTORY+1, :, :]
check("future frames == current frame repeated",
      torch.allclose(future_slice, current_obj.expand_as(future_slice)))
check("no NaN/Inf in full ohs",
      not torch.isnan(ohs).any() and not torch.isinf(ohs).any())


# ── 8. null_object gradient via absent slot ───────────────────────────────────
print()
print("=" * 60)
print("[8] null_object gradient — absent slot loss path")
print("=" * 60)

encoder.train()
if encoder.null_object.grad is not None:
    encoder.null_object.grad.zero_()

pres_mixed = mock_batch["object_presence"].clone()
pres_mixed[:, :, 0, :] = 1.0   # slot 0 present
pres_mixed[:, :, 1, :] = 0.0   # slot 1 absent
pres_mixed[:, :, 2, :] = 0.0   # slot 2 absent

bbox_r  = mock_batch["object_bbox"].clone()
state_r = mock_batch["object_state"].clone()
mask_r  = mock_batch["object_mask_crop"].clone()

out8 = encoder(pres_mixed, bbox_r, state_r, mask_r)

loss_absent = out8[:, :, 1:, :].sum()
loss_absent.backward()

check("null_object gradient non-zero from absent slot loss",
      encoder.null_object.grad is not None and encoder.null_object.grad.abs().sum() > 0,
      f"grad_norm={encoder.null_object.grad.norm().item():.4e}"
      if encoder.null_object.grad is not None else "grad=None")

check("encoder weights gradient from present slot",
      encoder.encoder[-1].weight.grad is not None
      and encoder.encoder[-1].weight.grad.abs().sum() > 0)


# ── 9. Non-zero init: presence=1 → non-zero token, sensitivity ───────────────
print()
print("=" * 60)
print("[9] Non-zero init verification")
print("=" * 60)

encoder.eval()

# Build a batch with presence=1, non-zero bbox and mask
pres9  = torch.zeros(B, F_CTX, N_OBJ, 1)
pres9[:, :, 0, :] = 1.0                         # slot 0 present, slots 1&2 absent

bbox9  = torch.rand(B, F_CTX, N_OBJ, 4) * 0.5 + 0.1   # non-zero bbox
mask9  = torch.rand(B, F_CTX, N_OBJ, CROP_SZ, CROP_SZ)
state9 = torch.zeros(B, F_CTX, N_OBJ, 1)

with torch.no_grad():
    out9 = encoder(pres9, bbox9, state9, mask9)

# presence=1 slot must produce non-zero token
slot0_norm = out9[:, :, 0, :].norm().item()
check("presence=1 slot output norm > 0",
      slot0_norm > 0,
      f"norm={slot0_norm:.4e}")

# presence=0 slot must equal null_object
null_bc9 = encoder.null_object.detach().view(1, 1, 1, -1).expand_as(out9)
check("presence=0 slot == null_object",
      torch.allclose(out9[:, :, 1:, :], null_bc9[:, :, 1:, :]))

# presence=1 token must differ from presence=0 (null_object) token
check("presence=1 token != null_object",
      not torch.allclose(out9[:, :, 0, :], null_bc9[:, :, 0, :]),
      f"slot0_norm={slot0_norm:.4e}  null_norm={encoder.null_object.norm().item():.4e}")

# changing bbox must change the output
bbox9_alt = bbox9.clone()
bbox9_alt[:, :, 0, :] = bbox9_alt[:, :, 0, :] + 0.2   # shift slot 0 bbox
with torch.no_grad():
    out9_alt = encoder(pres9, bbox9_alt, state9, mask9)
check("changing bbox changes output for present slot",
      not torch.allclose(out9[:, :, 0, :], out9_alt[:, :, 0, :]))

# changing mask must change the output
mask9_alt = mask9.clone()
mask9_alt[:, :, 0, :, :] = 1.0 - mask9[:, :, 0, :, :]   # invert slot 0 mask
with torch.no_grad():
    out9_mask = encoder(pres9, bbox9, state9, mask9_alt)
check("changing mask changes output for present slot",
      not torch.allclose(out9[:, :, 0, :], out9_mask[:, :, 0, :]))

# absent slots unchanged when bbox/mask of slot 0 changes
check("absent slots unaffected by slot 0 bbox change",
      torch.allclose(out9[:, :, 1:, :], out9_alt[:, :, 1:, :]))

print(f"  INFO  slot0 norm={slot0_norm:.4e}  "
      f"null_object norm={encoder.null_object.detach().norm().item():.4e}")


print()
print("=" * 60)
print("All tests PASSED." if all_pass else "SOME TESTS FAILED.")
print("=" * 60)
