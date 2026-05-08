"""
Phase 1 training: ObjectStateEncoder + mid_block_obj_adapter.

Goal: build and verify the object conditioning pathway in UNet.
      Phase 1 does NOT require the model to learn useful object semantics —
      that is Phase 2's job (presence/bbox/mask consistency + risk/rollback loss).

Freeze : VAE, image_encoder, text_encoder, action_encoder, UNet backbone.
Train  : ObjectStateEncoder (new, per-frame per-object),
         UNet.mid_block_obj_adapter (norm_h, q_proj, norm_obj, attn, out_proj, gate).

Forward:
  1. latent ← pre-encoded VAE latent (already in dataset)
  2. noise/timestep added (EDM schedule)
  3. action → action_encoder → action_hidden  [frozen]
  4. object fields (context frames only) → ObjectStateEncoder → object_hidden_states
  5. UNet(…, encoder_hidden_states=action_hidden, object_hidden_states=object_hidden_states)
  6. MSE loss on future frames only

Object conditioning rules:
  - Context frames  (UNet positions 0 … num_history)     → obj_encoded directly
  - Future frames   (UNet positions num_history+1 … end)  → repeat current-frame object state
  - obj_dropout % training dropout → object_hidden_states=None
  - object_hidden_states=None path is identical to original Ctrl-World baseline

Phase 1 success criteria (all 6 must hold before promoting checkpoint to Phase 2):
  1. ohs_norm > 0                   — object branch is alive
  2. obj_encoder grad > 0           — encoder receives gradient
  3. mid_block_obj_adapter grad > 0 — adapter receives gradient
  4. outproj_norm > 0               — out_proj weights are being updated
  5. UNet output changes with object input  (unet_diff_correct_vs_zero > 0)
  6. baseline_degradation small     — correct loss not much worse than zero loss

Phase 1 does NOT require:
  - correct < zero < shuffled < bbox_shifted  (that's Phase 2)
  - rollback recovery / object disappearance correction / risk-aware behavior

Usage:
  python scripts/train_phase1.py \\
    --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt \\
    --output_dir model_ckpt/phase1 \\
    --batch_size 2 --num_epochs 20 --lr 1e-4 --val_every 200
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import datetime
import torch
import torch.nn as nn
import einops
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger

from config import wm_args
from models.ctrl_world import CrtlWorld
from models.object_state_encoder import ObjectStateEncoder, N_OBJ, CROP_SZ
from dataset.dataset_droid_exp33 import Dataset_mix, TrackingDataset

logger = get_logger(__name__, log_level="INFO")


# ─── Debug helpers ────────────────────────────────────────────────────────────

def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm().item()
    return total


# ─── CLI args ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path",   type=str, required=True,
                   help="Base Ctrl-World checkpoint (.pt)")
    p.add_argument("--output_dir",  type=str, default=None)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int, default=2)
    p.add_argument("--num_epochs",  type=int, default=20)
    p.add_argument("--save_every",  type=int, default=500,
                   help="Checkpoint save interval (steps)")
    p.add_argument("--val_every",   type=int, default=200,
                   help="Validation interval (steps)")
    p.add_argument("--val_batches", type=int, default=4,
                   help="Number of batches to average for each val mode")
    p.add_argument("--obj_dropout", type=float, default=0.2,
                   help="Probability of setting object_hidden_states=None during training")
    p.add_argument("--tracking_root", type=str, default=None,
                   help="SAM3 tracking 폴더 경로 (지정 시 TrackingDataset과 혼합 학습)")
    p.add_argument("--tracking_only", action="store_true",
                   help="DROID 제외, tracking data만 사용 (debug용)")
    p.add_argument("--lr_enc",     type=float, default=None,
                   help="obj_encoder lr override (기본: --lr 값 사용)")
    p.add_argument("--lr_adapter", type=float, default=None,
                   help="mid_block_obj_adapter lr override (기본: --lr 값 사용)")
    p.add_argument("--val_tracking", action="store_true",
                   help="validation에 TrackingDataset 사용 (Phase1 health check용; --tracking_root 필요)")
    return p.parse_args()


# ─── Adapter checkpoint helpers ───────────────────────────────────────────────

def _save_state(model: CrtlWorld, obj_encoder: ObjectStateEncoder) -> dict:
    """Collect only trainable adapter weights for saving."""
    return {
        "obj_encoder":           obj_encoder.state_dict(),
        "mid_block_obj_adapter": model.unet.mid_block_obj_adapter.state_dict(),
    }


def _load_adapter_state(model: CrtlWorld, obj_encoder: ObjectStateEncoder,
                        path: str, device="cpu"):
    state = torch.load(path, map_location=device)
    obj_encoder.load_state_dict(state["obj_encoder"])
    model.unet.mid_block_obj_adapter.load_state_dict(state["mid_block_obj_adapter"])
    print(f"  [resume] loaded adapter from {path}")


# ─── Diffusion forward helpers ────────────────────────────────────────────────

def _build_input_latents(batch: dict, model: CrtlWorld, args, device, dtype,
                         cfg_dropout: bool = True):
    """
    Replicate the diffusion setup from CrtlWorld.forward (EDM schedule).

    cfg_dropout=False: action_hidden mask is all-True (no stochastic dropout).
                       Use this during validation so all 3 UNet forwards see
                       the same deterministic action conditioning.

    Returns: (input_latents, noisy_latents, c_noise, c_skip, c_out,
              loss_wt, action_hidden, added_time_ids, latents)
    """
    latents  = batch["latent"].to(device)              # (B, H+F, 4, h, w)
    texts    = batch["text"]
    action   = batch["action"].to(device)              # (B, H+F, 7)
    num_history = args.num_history
    bsz     = latents.shape[0]

    # ── current image condition (noisy) ─────────────────────────────────────
    current_img = latents[:, num_history, :]           # (B, 4, h, w)
    sigma_aug   = torch.rand([bsz, 1, 1, 1], device=device) * 0.2
    c_in_aug    = 1 / (sigma_aug ** 2 + 1) ** 0.5
    current_img = c_in_aug * (current_img + torch.randn_like(current_img) * sigma_aug)
    condition_latent = einops.repeat(
        current_img, "b c h w -> b f c h w",
        f=latents.shape[1]
    )
    if getattr(args, "his_cond_zero", False):
        condition_latent[:, :num_history] = 0.0

    # ── action encoding [frozen] ─────────────────────────────────────────────
    with torch.no_grad():
        action_hidden = model.action_encoder(
            action, texts, model.tokenizer, model.text_encoder,
            frame_level_cond=args.frame_level_cond,
        )  # (B, H+F, 1024)

    # CFG dropout (5%) — disabled during validation so action_hidden is fixed
    if cfg_dropout:
        uncond = torch.zeros_like(action_hidden)
        mask   = (torch.rand(bsz, device=device) > 0.05).view(bsz, 1, 1)
        action_hidden = action_hidden * mask + uncond * (~mask)

    # ── EDM noise schedule ───────────────────────────────────────────────────
    P_mean, P_std = 0.7, 1.6
    rnd     = torch.randn([bsz, 1, 1, 1, 1], device=device)
    sigma   = (rnd * P_std + P_mean).exp()
    c_skip  = 1 / (sigma ** 2 + 1)
    c_out   = -sigma / (sigma ** 2 + 1) ** 0.5
    c_in    = 1 / (sigma ** 2 + 1) ** 0.5
    c_noise = (sigma.log() / 4).reshape([bsz])
    loss_wt = (sigma ** 2 + 1) / sigma ** 2

    noisy_latents = latents + torch.randn_like(latents) * sigma

    # ── history noising ──────────────────────────────────────────────────────
    sigma_h    = torch.randn([bsz, num_history, 1, 1, 1], device=device) * 0.3
    history    = latents[:, :num_history]
    noisy_hist = (1 / (sigma_h ** 2 + 1) ** 0.5) * (
        history + sigma_h * torch.randn_like(history)
    )
    input_latents = torch.cat(
        [noisy_hist, c_in * noisy_latents[:, num_history:]], dim=1
    )
    # channel-wise stack with condition latent (SVD style)
    input_latents = torch.cat(
        [input_latents, condition_latent / model.vae.config.scaling_factor], dim=2
    )

    # ── added_time_ids ───────────────────────────────────────────────────────
    noise_aug_strength = 0.0
    added_time_ids = model.pipeline._get_add_time_ids(
        args.fps, args.motion_bucket_id, noise_aug_strength,
        action_hidden.dtype, bsz, 1, False,
    ).to(device)

    return (input_latents, noisy_latents, c_noise, c_skip, c_out,
            loss_wt, action_hidden, added_time_ids, latents)


def _encode_objects(obj_encoder: ObjectStateEncoder, batch: dict,
                    num_history: int, num_future: int,
                    device, dtype) -> torch.Tensor:
    """
    Encode context-frame object fields and map to UNet's full frame sequence.

    Context (F_ctx = num_history+1 frames):  UNet positions 0 … num_history
    Truly future (num_future-1 frames):       repeat last context (current frame)

    Returns: (B, num_history+num_future, N, 1024)
    """
    presence  = batch["object_presence"].to(device).to(dtype)   # (B, F_ctx, N, 1)
    bbox      = batch["object_bbox"].to(device).to(dtype)        # (B, F_ctx, N, 4)
    state_t   = batch["object_state"].to(device).to(dtype)      # (B, F_ctx, N, 1)
    mask_crop = batch["object_mask_crop"].to(device).to(dtype)   # (B, F_ctx, N, 16, 16)

    obj_encoded = obj_encoder(presence, bbox, state_t, mask_crop)  # (B, F_ctx, N, 1024)

    # Repeat current frame for truly-future UNet positions
    # F_ctx = num_history+1 covers positions 0..num_history
    # Positions num_history+1 .. num_history+num_future-1 need 4 more frames
    num_extra = num_future - 1  # frames beyond the current frame
    if num_extra > 0:
        current_obj = obj_encoded[:, -1:, :, :]                     # (B, 1, N, 1024)
        future_obj  = current_obj.expand(-1, num_extra, -1, -1)     # (B, extra, N, 1024)
        obj_encoded = torch.cat([obj_encoded, future_obj], dim=1)   # (B, H+F, N, 1024)

    return obj_encoded  # (B, num_history+num_future, N, 1024)


# ─── UNet forward from a pre-built diffusion pack ────────────────────────────

def _pred_from_pack(model: CrtlWorld, pack: tuple,
                    object_hidden_states, args) -> torch.Tensor:
    """Run UNet forward and return raw model_pred (no loss computation)."""
    (input_latents, noisy_latents, c_noise, c_skip, c_out,
     loss_wt, action_hidden, added_time_ids, latents) = pack
    return model.unet(
        input_latents, c_noise,
        encoder_hidden_states=action_hidden,
        added_time_ids=added_time_ids,
        frame_level_cond=args.frame_level_cond,
        object_hidden_states=object_hidden_states,
    ).sample  # (B, H+F, 4, h, w)


def _loss_from_pack(model: CrtlWorld, pack: tuple,
                    object_hidden_states, args) -> torch.Tensor:
    """
    Run UNet forward and compute MSE loss using an already-built diffusion pack.

    pack: output of _build_input_latents — all tensors are fixed across calls.
    object_hidden_states: (B, H+F, N, 1024) | None
    """
    (input_latents, noisy_latents, c_noise, c_skip, c_out,
     loss_wt, action_hidden, added_time_ids, latents) = pack

    model_pred = model.unet(
        input_latents,
        c_noise,
        encoder_hidden_states=action_hidden,
        added_time_ids=added_time_ids,
        frame_level_cond=args.frame_level_cond,
        object_hidden_states=object_hidden_states,
    ).sample  # (B, H+F, 4, h, w)

    predict_x0  = c_out * model_pred + c_skip * noisy_latents
    num_history = args.num_history
    future_pred  = predict_x0[:, num_history:]   # (B, F, 4, h, w)
    future_clean = latents[:, num_history:]       # clean target
    return ((future_pred - future_clean) ** 2 * loss_wt).mean()


# ─── Single training forward + loss ──────────────────────────────────────────

def _forward_loss(model: CrtlWorld, obj_encoder: ObjectStateEncoder,
                  batch: dict, args, device, dtype,
                  obj_hidden_override=None,
                  shuffle_obj: bool = False) -> torch.Tensor:
    """
    obj_hidden_override:
      None   → encode objects from batch (normal training path)
      "skip" → object_hidden_states=None  (baseline, CFG-dropout-style)
    shuffle_obj:
      True → shuffle batch dimension of object tokens (wrong context)
    """
    num_history = args.num_history
    num_future  = args.num_frames

    pack = _build_input_latents(batch, model, args, device, dtype, cfg_dropout=True)

    if obj_hidden_override == "skip":
        object_hidden_states = None
    else:
        object_hidden_states = _encode_objects(
            obj_encoder, batch, num_history, num_future, device, dtype
        )
        if shuffle_obj:
            idx = torch.randperm(object_hidden_states.shape[0], device=device)
            object_hidden_states = object_hidden_states[idx]

    return _loss_from_pack(model, pack, object_hidden_states, args)


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: CrtlWorld, obj_encoder: ObjectStateEncoder,
             val_loader, args, device, dtype, n_batches: int = 4):
    """
    Phase 1 validation — measures 6 health criteria, not loss ordering.

    Returns dict with keys:
      'correct', 'zero', 'shuffled', 'bbox_shifted'  — loss values
      'ohs_norm'     — object_hidden_states norm (criterion 1: must be > 0)
      'unet_diff'    — mean |pred_correct - pred_zero| (criterion 5: UNet responds to objects)
    """
    model.eval()
    obj_encoder.eval()

    totals = {
        "correct": 0.0, "zero": 0.0, "shuffled": 0.0, "bbox_shifted": 0.0,
        "presence_mean": 0.0, "present_ohs_norm": 0.0, "unet_diff": 0.0,
    }
    counted = 0

    _OBJ_KEYS = ("object_presence", "object_bbox", "object_state", "object_mask_crop")
    F_ctx = args.num_history + 1

    for batch in val_loader:
        if counted >= n_batches:
            break

        pack = _build_input_latents(batch, model, args, device, dtype, cfg_dropout=False)

        # correct: raw object fields → encode
        obj_enc = _encode_objects(
            obj_encoder, batch, args.num_history, args.num_frames, device, dtype
        )

        # criterion 1 (strengthened): presence_mean and present_ohs_norm
        pres = batch["object_presence"].to(device).float()   # (B, F_ctx, N, 1)
        pres_mask = (pres > 0.5).squeeze(-1)                 # (B, F_ctx, N) bool
        presence_mean = pres_mask.float().mean().item()
        ctx_enc       = obj_enc[:, :F_ctx, :, :]             # (B, F_ctx, N, 1024)
        present_tokens = ctx_enc[pres_mask]                  # (num_present, 1024)
        present_ohs_norm = (present_tokens.float().norm(dim=-1).mean().item()
                            if present_tokens.numel() > 0 else 0.0)

        # shuffled: batch-shuffle raw fields then re-encode (not just shuffle ohs)
        B_val = batch["object_presence"].shape[0]
        batch_shuf = {k: v for k, v in batch.items()}
        idx_cpu = torch.randperm(B_val)
        for fk in _OBJ_KEYS:
            t = batch_shuf[fk]
            batch_shuf[fk] = t[idx_cpu.to(t.device)]
        obj_shuf = _encode_objects(
            obj_encoder, batch_shuf, args.num_history, args.num_frames, device, dtype
        )

        # bbox_shifted: shift object_bbox then re-encode from scratch
        batch_bshift = {k: v for k, v in batch.items()}
        bbox = batch_bshift["object_bbox"].clone()
        shift = (torch.rand(2, device=bbox.device) - 0.5) * 0.6
        bbox[..., 0] = (bbox[..., 0] + shift[0]).clamp(0.0, 1.0)
        bbox[..., 2] = (bbox[..., 2] + shift[0]).clamp(0.0, 1.0)
        bbox[..., 1] = (bbox[..., 1] + shift[1]).clamp(0.0, 1.0)
        bbox[..., 3] = (bbox[..., 3] + shift[1]).clamp(0.0, 1.0)
        batch_bshift["object_bbox"] = bbox
        obj_bbox_shifted = _encode_objects(
            obj_encoder, batch_bshift, args.num_history, args.num_frames, device, dtype
        )

        # criterion 5: does UNet output actually change when object input changes?
        pred_correct = _pred_from_pack(model, pack, obj_enc, args)
        pred_zero    = _pred_from_pack(model, pack, None,    args)
        unet_diff    = (pred_correct - pred_zero).abs().mean().item()

        # diff diagnostics (first batch only)
        if counted == 0:
            diff_shuf = (obj_enc - obj_shuf).abs().mean().item()
            diff_bbox = (obj_enc - obj_bbox_shifted).abs().mean().item()
            print(
                f"\n[val debug]"
                f"  presence_mean={presence_mean:.4f}"
                f"  present_ohs_norm={present_ohs_norm:.4e}"
                f"\n  obj_enc.norm={obj_enc.norm().item():.4e}"
                f"  obj_shuf.norm={obj_shuf.norm().item():.4e}"
                f"  obj_bbox_shifted.norm={obj_bbox_shifted.norm().item():.4e}"
                f"\n  diff(enc,shuf)={diff_shuf:.4e}"
                f"  diff(enc,bbox_shifted)={diff_bbox:.4e}"
                f"  unet_diff(correct_vs_zero)={unet_diff:.4e}"
            )

        totals["correct"]          += _loss_from_pack(model, pack, obj_enc,         args).item()
        totals["zero"]             += _loss_from_pack(model, pack, None,             args).item()
        totals["shuffled"]         += _loss_from_pack(model, pack, obj_shuf,         args).item()
        totals["bbox_shifted"]     += _loss_from_pack(model, pack, obj_bbox_shifted, args).item()
        totals["presence_mean"]    += presence_mean
        totals["present_ohs_norm"] += present_ohs_norm
        totals["unet_diff"]        += unet_diff

        counted += 1

    model.train()
    obj_encoder.train()
    return {k: v / max(counted, 1) for k, v in totals.items()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cli = parse_args()
    args = wm_args()

    # Output dir
    if cli.output_dir:
        args.output_dir = cli.output_dir
    else:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        args.output_dir = f"model_ckpt/phase1_{ts}"

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    dtype  = torch.float16
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    # Disable old object encoder (use_object_state=False) — Phase 1 uses new encoder
    args.use_object_state = False
    args.use_warning      = False

    model = CrtlWorld(args)

    # Load base Ctrl-World checkpoint (action_encoder weights, etc.)
    missing, unexpected = model.load_state_dict(
        torch.load(cli.ckpt_path, map_location="cpu"), strict=False
    )

    # Allowed missing: new modules not present in the base checkpoint
    _allowed_missing = (
        "unet.mid_block_obj_adapter.",  # phase1 신규 adapter
        "obj_encoder.",                  # 별도 신규 모듈 (model state_dict에는 없음)
    )
    missing_allowed  = [k for k in missing if any(k.startswith(p) for p in _allowed_missing)]
    missing_backbone = [k for k in missing if not any(k.startswith(p) for p in _allowed_missing)]

    if missing_allowed:
        print(f"[INFO] expected missing (new modules): {len(missing_allowed)} keys "
              f"— mid_block_obj_adapter / obj_encoder will be randomly initialised")

    if missing_backbone:
        raise RuntimeError(
            f"[ERROR] backbone key(s) missing from checkpoint — "
            f"UNet / VAE / action_encoder weights not loaded.\n"
            f"  count : {len(missing_backbone)}\n"
            f"  sample: {missing_backbone[:8]}\n"
            f"Check that --ckpt_path points to a full Ctrl-World checkpoint."
        )

    # Allowed unexpected: old modules disabled by use_object_state=False / use_warning=False
    _allowed_unexpected = (
        "shape_projector.",       # old per-frame shape projector
        "object_state_encoder.",  # old object state encoder
        "warning_encoder.",       # old warning encoder
        "unet.shape_projector.",
        "unet.object_state_encoder.",
        "unet.warning_encoder.",
    )
    unexpected_known   = [k for k in unexpected
                          if any(k.startswith(p) for p in _allowed_unexpected)]
    unexpected_unknown = [k for k in unexpected
                          if not any(k.startswith(p) for p in _allowed_unexpected)]

    if unexpected_known:
        print(f"[INFO] ignoring {len(unexpected_known)} old module keys "
              f"(shape_projector / object_state_encoder / warning_encoder)")

    if unexpected_unknown:
        raise RuntimeError(
            f"[ERROR] unexpected key(s) in checkpoint not recognised by model.\n"
            f"  count : {len(unexpected_unknown)}\n"
            f"  sample: {unexpected_unknown[:8]}\n"
            f"Check that --ckpt_path is a Ctrl-World checkpoint for this architecture."
        )

    # ── Freeze everything, then unfreeze adapter ───────────────────────────────
    # VAE, image_encoder, text_encoder: already frozen in CrtlWorld.__init__
    # action_encoder: already frozen in CrtlWorld.__init__
    # UNet backbone: frozen in CrtlWorld.__init__, BUT mid_block_obj_adapter is inside UNet
    for p in model.unet.mid_block_obj_adapter.parameters():
        p.requires_grad_(True)

    # Freeze audit: inside CrtlWorld, ONLY mid_block_obj_adapter.* must be trainable
    unexpected_trainable = [
        name for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("unet.mid_block_obj_adapter.")
    ]
    if unexpected_trainable:
        raise RuntimeError(
            f"[ERROR] freeze audit failed — unexpected trainable params in model "
            f"(expected only unet.mid_block_obj_adapter.*):\n"
            + "\n".join(f"  {n}" for n in unexpected_trainable[:10])
        )
    print(f"[INFO] freeze audit passed — only unet.mid_block_obj_adapter.* trainable in model")

    # New per-frame ObjectStateEncoder (always trainable)
    obj_encoder = ObjectStateEncoder(n_objects=N_OBJ, crop_sz=CROP_SZ)

    # Param count
    n_adapter = sum(p.numel() for p in model.unet.mid_block_obj_adapter.parameters())
    n_enc     = sum(p.numel() for p in obj_encoder.parameters())
    print(f"[INFO] Trainable: mid_block_obj_adapter={n_adapter/1e6:.2f}M  "
          f"obj_encoder={n_enc/1e6:.2f}M  "
          f"total={( n_adapter+n_enc)/1e6:.2f}M")

    # ── Optimizer (per-group lr) ───────────────────────────────────────────────
    lr_adapter = cli.lr_adapter if cli.lr_adapter is not None else cli.lr
    lr_enc     = cli.lr_enc     if cli.lr_enc     is not None else cli.lr
    param_groups = [
        {"params": list(model.unet.mid_block_obj_adapter.parameters()),
         "lr": lr_adapter, "name": "mid_block_obj_adapter"},
        {"params": list(obj_encoder.parameters()),
         "lr": lr_enc, "name": "obj_encoder"},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=cli.weight_decay
    )
    print(f"[INFO] lr: mid_block_obj_adapter={lr_adapter:.2e}  obj_encoder={lr_enc:.2e}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if cli.val_tracking:
        if not cli.tracking_root:
            raise ValueError("--val_tracking requires --tracking_root")
        val_dataset = TrackingDataset(cli.tracking_root, args)
        print(f"[INFO] val_dataset = TrackingDataset  ({len(val_dataset)} samples)")
    else:
        val_dataset = Dataset_mix(args, mode="val")

    if cli.tracking_only:
        if not cli.tracking_root:
            raise ValueError("--tracking_only requires --tracking_root")
        train_dataset = TrackingDataset(cli.tracking_root, args)
        print(f"[INFO] tracking_only: {len(train_dataset)} samples")
    else:
        train_dataset = Dataset_mix(args, mode="train")
        if cli.tracking_root:
            tracking_train = TrackingDataset(cli.tracking_root, args)
            train_dataset  = torch.utils.data.ConcatDataset([train_dataset, tracking_train])
            print(f"[INFO] ConcatDataset: DROID + Tracking "
                  f"(total {len(train_dataset)} train samples)")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cli.batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cli.batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    # ── Accelerate prepare ────────────────────────────────────────────────────
    model, obj_encoder, optimizer, train_loader = accelerator.prepare(
        model, obj_encoder, optimizer, train_loader
    )
    # Re-collect after prepare so clip_grad_norm_ targets the (DDP-wrapped) params
    _um_post = accelerator.unwrap_model(model)
    _ue_post = accelerator.unwrap_model(obj_encoder)
    trainable_params = (
        list(_um_post.unet.mid_block_obj_adapter.parameters()) +
        list(_ue_post.parameters())
    )
    model.train()
    obj_encoder.train()

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(cli.num_epochs):
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch+1}/{cli.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in pbar:
            with accelerator.accumulate(model, obj_encoder):

                # dropout=0.0이면 항상 object branch ON (debug 권장)
                use_obj = torch.rand(1).item() >= cli.obj_dropout

                with accelerator.autocast():
                    loss = _forward_loss(
                        model,
                        obj_encoder,
                        batch, args, device, dtype,
                        obj_hidden_override=(None if use_obj else "skip"),
                    )

                accelerator.backward(loss)

                # ── Debug log every 20 steps (before clip/step) ───────────
                if accelerator.is_main_process and (global_step + 1) % 20 == 0:
                    _umodel   = accelerator.unwrap_model(model)
                    _uenc     = accelerator.unwrap_model(obj_encoder)
                    adapter   = _umodel.unet.mid_block_obj_adapter

                    gate_sig  = torch.sigmoid(adapter.gate).item()
                    outproj_n = adapter.out_proj.weight.detach().float().norm().item()

                    gn_enc     = grad_norm(_uenc.parameters())
                    gn_adapter = grad_norm(adapter.parameters())
                    gn_outproj = grad_norm(adapter.out_proj.parameters())

                    # object_hidden_states stats from current batch
                    with torch.no_grad():
                        _ohs = _encode_objects(
                            _uenc, batch,
                            args.num_history, args.num_frames,
                            device, torch.float32,
                        )
                    ohs_norm = _ohs.float().norm().item()
                    ohs_mean = _ohs.float().mean().item()
                    ohs_std  = _ohs.float().std().item()

                    pres_mean = batch["object_presence"].float().mean().item()
                    mask_sum  = batch["object_mask_crop"].float().sum().item()

                    print(
                        f"\n[debug step={global_step+1}]"
                        f"  gate_sig={gate_sig:.6f}"
                        f"  outproj_norm={outproj_n:.6e}"
                        f"\n  grad: enc={gn_enc:.4e}"
                        f"  adapter={gn_adapter:.4e}"
                        f"  outproj={gn_outproj:.4e}"
                        f"\n  ohs: shape={tuple(_ohs.shape)}"
                        f"  norm={ohs_norm:.4e}"
                        f"  mean={ohs_mean:.4e}"
                        f"  std={ohs_std:.4e}"
                        f"\n  batch: presence_mean={pres_mean:.4f}"
                        f"  mask_sum={mask_sum:.1f}"
                    )

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    obj="ON" if use_obj else "OFF",
                    step=global_step,
                )

            # ── Validation (barrier: all processes sync) ───────────────
            if global_step % cli.val_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    val = validate(
                        accelerator.unwrap_model(model),
                        accelerator.unwrap_model(obj_encoder),
                        val_loader, args, device, torch.float32,
                        n_batches=cli.val_batches,
                    )
                    # baseline_degradation: positive = correct WORSE than zero
                    degradation       = val["correct"] - val["zero"]
                    degradation_ratio = max(degradation, 0.0) / max(val["zero"], 1e-8)
                    sc = val["shuffled"]     - val["correct"]
                    bc = val["bbox_shifted"] - val["correct"]

                    # Phase1 health flags
                    # c1: presence data exists AND present-slot encodings are non-trivial
                    c1 = (val["presence_mean"]    > 0.01 and
                          val["present_ohs_norm"] > 1e-3)
                    c5 = val["unet_diff"] > 1e-5          # UNet responds to object input
                    c6 = degradation_ratio < 0.05          # degradation < 5% of baseline

                    print(
                        f"\n[val step={global_step}]"
                        f"\n  correct     ={val['correct']:.8f}"
                        f"\n  zero        ={val['zero']:.8f}"
                        f"  (degradation={degradation:+.3e}  ratio={degradation_ratio:+.4f})"
                        f"\n  shuffled    ={val['shuffled']:.8f}  (shuf-correct={sc:+.3e})"
                        f"\n  bbox_shifted={val['bbox_shifted']:.8f}  (bbox-correct={bc:+.3e})"
                        f"\n  presence_mean={val['presence_mean']:.4f}"
                        f"  present_ohs_norm={val['present_ohs_norm']:.4e}"
                        f"  unet_diff={val['unet_diff']:.4e}"
                        f"\n  Phase1: [{'✓' if c1 else '✗'}]ohs_alive"
                        f"  [{'✓' if c5 else '✗'}]unet_sensitive"
                        f"  [{'✓' if c6 else '✗'}]degradation_ok(<5%)"
                        f"  (target: all ✓)"
                    )
                accelerator.wait_for_everyone()

            # ── Checkpoint (barrier: all processes sync) ───────────────
            if global_step % cli.save_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}.pt"
                    )
                    accelerator.save(
                        _save_state(
                            accelerator.unwrap_model(model),
                            accelerator.unwrap_model(obj_encoder),
                        ),
                        ckpt_path,
                    )
                    print(f"  saved → {ckpt_path}")
                accelerator.wait_for_everyone()

    # ── Final save ────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoint-final.pt")
        accelerator.save(
            _save_state(
                accelerator.unwrap_model(model),
                accelerator.unwrap_model(obj_encoder),
            ),
            final_path,
        )
        print(f"\n완료: {final_path}")


if __name__ == "__main__":
    main()
