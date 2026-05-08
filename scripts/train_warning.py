"""
Warning-conditioned Ctrl-World fine-tuning.

4가지 실험 모드:
  baseline : 기존 Ctrl-World만
  obj      : obj_state + obj_shape (warning 없음)
  hard     : obj + hard warning (crushed/vanished)
  full     : obj + hard + soft warning (occluded 포함)

Usage:
  python scripts/train_warning.py \\
    --tracking_root /path/to/tracking_data \\
    --mode full \\
    --output_dir model_ckpt/warning_full \\
    --ckpt_path /path/to/base.pt

python scripts/train_warning.py \
  --tracking_root /home/dgu/minyoung/droid_data/tracking \
  --mode obj \
  --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt \
  --batch_size 1 \
  --num_epochs 3 \
  --lr 1e-4 \
  --save_every 50
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import datetime
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from config import wm_args
from models.ctrl_world import CrtlWorld
from dataset.dataset_warning import WarningDataset, HISTORY_LEN, PRED_LEN

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_root', type=str, required=True,
                        help='에피소드별 tracking.json + latent.pt가 있는 루트 디렉토리')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['baseline', 'obj', 'hard', 'full'],
                        help='실험 모드')
    parser.add_argument('--ckpt_path',  type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--save_every', type=int, default=500,
                        help='몇 step마다 checkpoint 저장')
    parser.add_argument('--obj_scale', type=float, default=0.1,
                        help='obj token residual injection scale (0.0=no injection)')
    parser.add_argument('--warn_scale', type=float, default=0.1,
                        help='warning token residual injection scale')
    return parser.parse_args()


def _adapter_state(model) -> dict:
    """adapter 모듈만 flat state_dict으로 반환. load_state_dict(strict=False) 호환."""
    state = {}
    for name, module in [
        ('shape_projector',      model.shape_projector),
        ('object_state_encoder', model.object_state_encoder),
        ('warning_encoder',      model.warning_encoder),
    ]:
        if module is not None:
            for k, v in module.state_dict().items():
                state[f'{name}.{k}'] = v
    return state


def main():
    args_cli = parse_args()
    args = wm_args()

    # 모드에 따라 conditioning 플래그 설정
    args.use_object_state = (args_cli.mode != 'baseline')
    args.use_warning      = (args_cli.mode in ('hard', 'full'))
    args.num_history      = HISTORY_LEN
    args.num_frames       = PRED_LEN

    args.obj_injection_scale     = args_cli.obj_scale
    args.warning_injection_scale = args_cli.warn_scale

    if args_cli.ckpt_path:
        args.ckpt_path = args_cli.ckpt_path
    if args_cli.output_dir:
        args.output_dir = args_cli.output_dir
    else:
        now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        scale_tag = f"scale{args_cli.obj_scale:.2f}"
        args.output_dir = f"model_ckpt/{scale_tag}_warning_{args_cli.mode}_{now}"

    accelerator = Accelerator(mixed_precision='fp16')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 모델 ────────────────────────────────────────────────
    model = CrtlWorld(args)
    if args_cli.ckpt_path:
        missing, unexpected = model.load_state_dict(
            torch.load(args_cli.ckpt_path, map_location='cpu'), strict=False)
        _adapter_prefixes = ('shape_projector.', 'object_state_encoder.', 'warning_encoder.')
        missing_adapter  = [k for k in missing if any(k.startswith(p) for p in _adapter_prefixes)]
        missing_backbone = [k for k in missing if not any(k.startswith(p) for p in _adapter_prefixes)]
        if missing_adapter:
            print(f"[INFO] adapter modules not in base ckpt → zero-init으로 학습 시작 (정상): "
                  f"{set(k.split('.')[0] for k in missing_adapter)}")
        if missing_backbone:
            print(f"[WARN] unexpected missing backbone keys: {missing_backbone}")

    # ── trainable params ─────────────────────────────────────
    trainable = []
    for name, module in [
        ('shape_projector',       model.shape_projector),
        ('object_state_encoder',  model.object_state_encoder),
        ('warning_encoder',       model.warning_encoder),
    ]:
        if module is not None:
            params = list(module.parameters())
            trainable += params
            n = sum(p.numel() for p in params)
            print(f"  trainable {name}: {n/1e3:.1f}K params")

    assert trainable, "trainable 파라미터가 없습니다. 모드 설정을 확인하세요."

    optimizer = torch.optim.AdamW(
        trainable, lr=args_cli.lr, weight_decay=args_cli.weight_decay)

    # ── 데이터셋 ─────────────────────────────────────────────
    dataset = WarningDataset(
        tracking_root=args_cli.tracking_root,
        mode=args_cli.mode,
        history_len=HISTORY_LEN,
        pred_len=PRED_LEN,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args_cli.batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    global_step = 0
    for epoch in range(args_cli.num_epochs):
        pbar = tqdm(loader, desc=f"[{args_cli.mode}] epoch {epoch+1}/{args_cli.num_epochs}",
                    disable=not accelerator.is_main_process)
        for batch in pbar:
            with accelerator.accumulate(model):
                loss, _ = model(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

                if global_step % args_cli.save_every == 0:
                    scale_tag = f"scale{args_cli.obj_scale:.2f}"
                    ckpt = os.path.join(args.output_dir,
                                        f"{scale_tag}_checkpoint-{global_step}.pt")
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(_adapter_state(unwrapped), ckpt)
                    print(f"\n  saved → {ckpt}")

    if accelerator.is_main_process:
        scale_tag = f"scale{args_cli.obj_scale:.2f}"
        final = os.path.join(args.output_dir, f"{scale_tag}_checkpoint-final.pt")
        torch.save(_adapter_state(accelerator.unwrap_model(model)), final)
        print(f"\n완료: {final}")


if __name__ == '__main__':
    main()
