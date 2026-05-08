"""
debug_gripper_scale.py
─────────────────────
gripper scale mismatch 진단 스크립트.

목적:
  droid_tracking annotation의 gripper 값이 Ctrl-World가 학습한
  droid_subset/stat.json 기준과 같은 스케일인지 확인한다.

분석 (--analyze_only):
  1. states[:,6] 이 observation.state.gripper_position vs
     action.gripper_position 중 어느 쪽과 일치하는지 확인
  2. 에피소드별 raw/normalized gripper 곡선 플롯
  3. droid_subset 샘플과 통계 비교

실험 (--run_rollout):
  WM forward 1회 (pi0 없음, GT action_cond 사용):
    - original:       annotation 그대로
    - force_close:    grasp window에서 gripper → 1.0
    - rescale_06112:  ep85 한정 gripper / 0.6112

출력:
  debug_gripper_scale/
    analysis/
      gripper_field_source.json
      gripper_curves_raw.png
      gripper_curves_norm.png
    episode_{xxx}/
      {original,force_close,rescale_06112}/
        action_cond_summary.json
        rollout.mp4

사용 예:
  # 분석만
  python scripts/debug_gripper_scale.py --analyze_only \
      --episodes 85 118 \
      --val_dataset_dir /home/dgu/minyoung/droid_data/tracking/

  # 분석 + WM rollout 실험
  CUDA_VISIBLE_DEVICES=0 python scripts/debug_gripper_scale.py \
      --episodes 85 \
      --val_dataset_dir /home/dgu/minyoung/droid_data/tracking/ \
      --val_skip 1 \
      --run_rollout \
      --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt \
      --svd_model_path checkpoints/svd \
      --start_idx 0
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── stat.json 로드 ──────────────────────────────────────────────────────────
STAT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset_meta_info/droid/stat.json",
)


def load_stat():
    with open(STAT_PATH) as f:
        d = json.load(f)
    return np.array(d['state_01']), np.array(d['state_99'])


def normalize_bound(data, p01, p99, eps=1e-8):
    return np.clip(2 * (data - p01) / (p99 - p01 + eps) - 1, -1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def load_episode_anno(val_dataset_dir, ep_id):
    """annotation.json 로드 (droid_tracking 폴더 구조 대응)."""
    # droid_tracking: episode_XXXXXX/annotation.json
    ep_dir = os.path.join(val_dataset_dir, f"episode_{ep_id:06d}")
    ann_path = os.path.join(ep_dir, "annotation.json")
    if not os.path.exists(ann_path):
        # fallback: val_dataset_dir/annotation/val/<id>.json (droid_subset)
        ann_path = os.path.join(val_dataset_dir, "annotation", "val", f"{ep_id}.json")
    with open(ann_path) as f:
        return json.load(f)


def check_field_source(anno):
    """
    states[:,6] 이 obs.state.gripper 인지 action.gripper 인지 확인.
    returns: dict with 'states_g', 'obs_g', 'action_g', 'source'
    """
    states = np.array(anno['states'])               # (T, 7)
    g_states = states[:, 6]

    g_obs    = np.array(anno.get('observation.state.gripper_position', [[]])).ravel()
    g_action = np.array(anno.get('action.gripper_position', [[]])).ravel()

    # length 맞추기
    T = len(g_states)
    g_obs    = g_obs[:T]    if len(g_obs)    >= T else g_obs
    g_action = g_action[:T] if len(g_action) >= T else g_action

    def match_score(a, b):
        n = min(len(a), len(b))
        if n == 0:
            return float('nan')
        return float(np.mean(np.abs(a[:n] - b[:n])))

    mae_obs    = match_score(g_states, g_obs)
    mae_action = match_score(g_states, g_action)

    if np.isnan(mae_obs) and np.isnan(mae_action):
        source = 'unknown'
    elif np.isnan(mae_obs):
        source = 'action'
    elif np.isnan(mae_action):
        source = 'obs'
    elif mae_obs < mae_action:
        source = 'obs'
    else:
        source = 'action'

    return {
        'states_g':   g_states,
        'obs_g':      g_obs,
        'action_g':   g_action,
        'source':     source,
        'mae_obs':    mae_obs,
        'mae_action': mae_action,
    }


def analyze_episodes(val_dataset_dir, episode_ids, out_dir, subset_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    p01, p99 = load_stat()

    field_source_report = {}
    fig_raw, axes_raw   = plt.subplots(len(episode_ids), 1,
                                        figsize=(12, 3 * len(episode_ids)))
    fig_norm, axes_norm = plt.subplots(len(episode_ids), 1,
                                        figsize=(12, 3 * len(episode_ids)))
    if len(episode_ids) == 1:
        axes_raw  = [axes_raw]
        axes_norm = [axes_norm]

    for ax_r, ax_n, ep_id in zip(axes_raw, axes_norm, episode_ids):
        try:
            anno = load_episode_anno(val_dataset_dir, ep_id)
        except FileNotFoundError as e:
            print(f"[WARN] ep{ep_id}: {e}")
            continue

        info = check_field_source(anno)
        states = np.array(anno['states'])
        g_states = states[:, 6]
        g_norm   = normalize_bound(g_states, p01[6], p99[6])

        print(f"\n── episode {ep_id:06d} ──────────────────────────────")
        print(f"  states[:,6]  source  : {info['source']}")
        print(f"  mae_obs={info['mae_obs']:.6f}  mae_action={info['mae_action']:.6f}")
        print(f"  states[:,6]  raw : min={g_states.min():.4f}  max={g_states.max():.4f}")
        if len(info['obs_g']):
            print(f"  obs.gripper  raw : min={info['obs_g'].min():.4f}  max={info['obs_g'].max():.4f}")
        if len(info['action_g']):
            print(f"  action.grip raw  : min={info['action_g'].min():.4f}  max={info['action_g'].max():.4f}")
        print(f"  normalized   : min={g_norm.min():.4f}  max={g_norm.max():.4f}")

        field_source_report[str(ep_id)] = {
            'source':     info['source'],
            'mae_obs':    info['mae_obs'],
            'mae_action': info['mae_action'],
            'raw_min':    float(g_states.min()),
            'raw_max':    float(g_states.max()),
            'norm_min':   float(g_norm.min()),
            'norm_max':   float(g_norm.max()),
            'stat_p01':   float(p01[6]),
            'stat_p99':   float(p99[6]),
        }

        # raw 곡선
        T = len(g_states)
        ax_r.plot(g_states, label='states[:,6]')
        if len(info['obs_g']):
            ax_r.plot(info['obs_g'][:T], '--', label='obs.state.gripper', alpha=0.7)
        if len(info['action_g']):
            ax_r.plot(info['action_g'][:T], ':', label='action.gripper', alpha=0.7)
        ax_r.axhline(p99[6], color='red', linestyle='-.', linewidth=0.8, label=f'stat p99={p99[6]:.4f}')
        ax_r.set_title(f'ep{ep_id:06d}  raw gripper  [source={info["source"]}]')
        ax_r.set_ylabel('raw value')
        ax_r.legend(fontsize=8)
        ax_r.set_ylim(-0.05, 1.1)

        # normalized 곡선
        ax_n.plot(g_norm, label='states[:,6] normalized')
        ax_n.axhline(1.0, color='red', linestyle='-.', linewidth=0.8, label='full-close norm=+1.0')
        ax_n.set_title(f'ep{ep_id:06d}  normalized gripper')
        ax_n.set_ylabel('normalized [-1,1]')
        ax_n.legend(fontsize=8)
        ax_n.set_ylim(-1.1, 1.1)

    # droid_subset 비교 (있을 경우)
    if subset_dir:
        subset_files = glob.glob(os.path.join(subset_dir, 'annotation', 'val', '*.json'))[:20]
        subset_maxes = []
        for sf in subset_files:
            with open(sf) as f:
                sd = json.load(f)
            ss = np.array(sd.get('states', []))
            if ss.ndim == 2 and ss.shape[1] >= 7:
                subset_maxes.append(float(ss[:, 6].max()))
        if subset_maxes:
            print(f"\n── droid_subset sample gripper raw max ──────────────")
            print(f"  n={len(subset_maxes)}  min={min(subset_maxes):.4f}  "
                  f"max={max(subset_maxes):.4f}  "
                  f"mean={np.mean(subset_maxes):.4f}")
            field_source_report['subset_gripper_max_samples'] = subset_maxes

    fig_raw.tight_layout()
    fig_raw.savefig(os.path.join(out_dir, 'gripper_curves_raw.png'), dpi=120)
    plt.close(fig_raw)

    fig_norm.tight_layout()
    fig_norm.savefig(os.path.join(out_dir, 'gripper_curves_norm.png'), dpi=120)
    plt.close(fig_norm)

    out_json = os.path.join(out_dir, 'gripper_field_source.json')
    with open(out_json, 'w') as f:
        json.dump(field_source_report, f, indent=2)

    print(f"\n[ANALYSIS] saved → {out_dir}")
    return field_source_report


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: LIGHTWEIGHT AGENT (pi0 없음, Ctrl-World만)
# ═══════════════════════════════════════════════════════════════════════════

def build_agent(args):
    """pi0 없이 Ctrl-World만 로드하는 경량 agent."""
    import torch
    from accelerate import Accelerator
    from models.ctrl_world import CrtlWorld

    class LightAgent:
        def __init__(self, args):
            self.args        = args
            self.accelerator = Accelerator()
            self.device      = self.accelerator.device
            self.dtype       = args.dtype

            self.model = CrtlWorld(args)
            missing, unexpected = self.model.load_state_dict(
                torch.load(args.val_model_path, map_location='cpu'), strict=False
            )
            # adapter keys missing → expected when no --adapter_ckpt
            backbone_miss = [k for k in missing if not k.startswith(
                ('shape_projector.', 'object_state_encoder.', 'warning_encoder.'))]
            if backbone_miss:
                print(f"[WARN] missing backbone keys: {backbone_miss}")
            self.model.to(self.device).to(self.dtype)
            self.model.eval()

            with open(args.data_stat_path) as f:
                stat = json.load(f)
            self.state_p01 = np.array(stat['state_01'])[None, :]
            self.state_p99 = np.array(stat['state_99'])[None, :]

        def normalize_bound(self, data, dmin, dmax, eps=1e-8):
            return np.clip(2 * (data - dmin) / (dmax - dmin + eps) - 1, -1, 1)

        def encode_frames(self, frames_uint8):
            """(T, H, W, 3) uint8 → (T, 4, h, w) VAE latent"""
            import torch
            vae    = self.model.pipeline.vae
            device = self.device
            dtype  = self.dtype
            T = frames_uint8.shape[0]
            x = torch.from_numpy(frames_uint8).to(dtype).to(device)
            x = x.permute(0, 3, 1, 2) / 255.0 * 2 - 1
            latents = []
            with torch.no_grad():
                for i in range(0, T, 4):
                    latents.append(
                        vae.encode(x[i:i+4]).latent_dist.sample()
                        .mul_(vae.config.scaling_factor)
                    )
            return torch.cat(latents, dim=0)

    agent_obj = LightAgent(args)
    return agent_obj


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════

def find_grasp_start(gripper_raw, threshold=0.05):
    """gripper가 처음으로 threshold 이상 올라가는 프레임 index."""
    for i, g in enumerate(gripper_raw):
        if g > threshold:
            return i
    return len(gripper_raw)  # 한 번도 안 닫힘


def patch_gripper(action_cond_raw, mode, ep_id, p01, p99):
    """
    action_cond_raw: (T, 7) raw (비정규화)
    returns: (T, 7) raw, patched
    """
    g = action_cond_raw[:, 6].copy()
    T = len(g)

    if mode == 'original':
        pass

    elif mode == 'force_close':
        grasp_start = find_grasp_start(g)
        print(f"  [force_close] grasp_start={grasp_start}/{T}")
        if grasp_start < T:
            g[grasp_start:] = p99[6]      # stat p99 ≈ 1.0 → normalized +1.0
        else:
            # 한 번도 안 닫힌 경우: 마지막 1/3을 강제 close
            g[T * 2 // 3:] = p99[6]
            print(f"  [force_close] no grasp detected → forcing last 1/3")

    elif mode == 'rescale_06112':
        if ep_id == 85:
            g = np.clip(g / 0.6112, 0, 1)
            print(f"  [rescale_06112] applied (ep85 only), new max={g.max():.4f}")
        else:
            print(f"  [rescale_06112] skipped (ep{ep_id} ≠ 85)")

    patched = action_cond_raw.copy()
    patched[:, 6] = g
    return patched


def load_video(video_path, frame_ids):
    """mp4에서 특정 프레임들 로드 → (T, H, W, 3) uint8"""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    try:
        all_frames = vr.get_batch(range(len(vr))).asnumpy()
    except:
        all_frames = vr.get_batch(range(len(vr))).numpy()
    ids = np.clip(frame_ids, 0, len(all_frames) - 1)
    return all_frames[ids]


def run_experiments(agent, val_dataset_dir, ep_id, start_idx, pred_step,
                    num_history, num_frames, val_skip, out_root, modes, view_idx=0):
    import torch
    import mediapy

    p01 = agent.state_p01[0]  # (7,)
    p99 = agent.state_p99[0]  # (7,)

    # ── annotation 로드 ─────────────────────────────────────────────
    anno = load_episode_anno(val_dataset_dir, ep_id)
    length      = anno.get('video_length', len(anno['states']))
    instruction = anno.get('texts', ['unknown'])[0]
    states_raw  = np.array(anno['states'])   # (video_length, 7) raw

    total_steps = pred_step + num_history + 8
    frame_ids = np.arange(start_idx, start_idx + total_steps * val_skip, val_skip)
    frame_ids = np.clip(frame_ids, 0, length - 1).astype(int)
    print(f"\nep{ep_id:06d}: length={length}  frame_ids[0:5]={frame_ids[:5]}")

    action_cond_raw = states_raw[frame_ids]   # (total_steps, 7)

    # ── 비디오 로드 ──────────────────────────────────────────────────
    videos_by_view = anno.get('videos', [])
    # droid_tracking: videos 항목이 상대경로 또는 파일명
    ep_dir = os.path.join(val_dataset_dir, f"episode_{ep_id:06d}")
    vid_paths = []
    for vid_name in videos_by_view:
        p = vid_name if os.path.isabs(vid_name) else os.path.join(ep_dir, vid_name)
        if os.path.exists(p):
            vid_paths.append(p)
    if not vid_paths:
        # fallback: episode_XXX_0.mp4, 1.mp4, 2.mp4
        for v_id in range(3):
            p = os.path.join(ep_dir, f"episode_{ep_id:06d}_{v_id}.mp4")
            if os.path.exists(p):
                vid_paths.append(p)
    if not vid_paths:
        print(f"[ERROR] no video found in {ep_dir}")
        return

    # VAE encode
    all_latents = []
    for vp in vid_paths:
        frames = load_video(vp, frame_ids)   # (T, H, W, 3)
        lat    = agent.encode_frames(frames) # (T, 4, h, w)
        all_latents.append((frames, lat))
    print(f"  loaded {len(all_latents)} views, latent shape: {all_latents[0][1].shape}")

    primary_frames  = all_latents[view_idx][0]  # (T, H, W, 3)
    primary_latents = all_latents[view_idx][1]  # (T, 4, h, w)

    # history: 첫 num_history 프레임 latents
    his_latents = primary_latents[:num_history].unsqueeze(0)   # (1, H, 4, h, w)
    cond_latent = primary_latents[num_history - 1]             # (4, h, w)

    # GT latent for forward_wm
    gt_latents  = primary_latents[num_history: num_history + pred_step]  # (pred_step, 4, h, w)

    # ── 실험별 loop ─────────────────────────────────────────────────
    for mode in modes:
        print(f"\n  ── mode={mode} ─────────────────────")
        exp_dir = os.path.join(out_root, f"episode_{ep_id:06d}", mode)
        os.makedirs(exp_dir, exist_ok=True)

        patched_raw = patch_gripper(action_cond_raw.copy(), mode, ep_id, p01, p99)
        action_cond_norm = agent.normalize_bound(patched_raw, p01, p99)   # (T, 7)

        # summary JSON
        g_raw  = patched_raw[:, 6]
        g_norm = action_cond_norm[:, 6]
        summary = {
            'mode':       mode,
            'episode':    ep_id,
            'start_idx':  int(start_idx),
            'instruction': instruction,
            'gripper_raw_min':  float(g_raw.min()),
            'gripper_raw_max':  float(g_raw.max()),
            'gripper_norm_min': float(g_norm.min()),
            'gripper_norm_max': float(g_norm.max()),
            'grasp_start_frame': int(find_grasp_start(g_raw)),
            'stat_p01_6': float(p01[6]),
            'stat_p99_6': float(p99[6]),
        }
        with open(os.path.join(exp_dir, 'action_cond_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  gripper norm: min={g_norm.min():.4f}  max={g_norm.max():.4f}")

        # 그리퍼 곡선 저장
        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        axes[0].plot(g_raw)
        axes[0].axhline(p99[6], color='red', linestyle='--', label=f'p99={p99[6]:.4f}')
        axes[0].set_title(f'ep{ep_id} [{mode}] gripper raw')
        axes[0].legend()
        axes[1].plot(g_norm)
        axes[1].axhline(1.0, color='red', linestyle='--', label='+1.0 (full close)')
        axes[1].set_ylim(-1.1, 1.1)
        axes[1].set_title(f'ep{ep_id} [{mode}] gripper normalized')
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(exp_dir, 'gripper_curve.png'), dpi=100)
        plt.close(fig)

        # ── World Model forward ──────────────────────────────────
        args = agent.args
        import torch as _torch

        ac_tensor = _torch.tensor(action_cond_norm).unsqueeze(0).to(agent.device).to(agent.dtype)
        # action_cond shape: (1, num_history + pred_step, 7)
        # 필요한 길이만큼 자르기
        needed = args.num_frames + args.num_history
        if ac_tensor.shape[1] < needed:
            pad = needed - ac_tensor.shape[1]
            ac_tensor = _torch.cat([
                ac_tensor,
                ac_tensor[:, -1:].expand(-1, pad, -1)
            ], dim=1)
        ac_tensor = ac_tensor[:, :needed]

        pipeline = agent.model.pipeline
        with _torch.no_grad():
            output = pipeline(
                image=cond_latent.unsqueeze(0).to(agent.device).to(agent.dtype),
                image_latents=his_latents.to(agent.device).to(agent.dtype),
                action=ac_tensor,
                num_frames=args.num_frames,
                num_inference_steps=getattr(args, 'num_inference_steps', 25),
                decode_chunk_size=getattr(args, 'decode_chunk_size', 8),
                fps=args.fps,
                motion_bucket_id=getattr(args, 'motion_bucket_id', 127),
                noise_aug_strength=getattr(args, 'noise_aug_strength', 0.02),
                output_type='pt',
                generator=_torch.manual_seed(getattr(args, 'seed', 42)),
            )

        # decode
        vae = pipeline.vae
        pred_latents = output.frames[0]  # (pred_step, 4, h, w)
        with _torch.no_grad():
            decoded = vae.decode(
                pred_latents.to(agent.device).to(agent.dtype) / vae.config.scaling_factor
            ).sample
        # (T, 3, H, W) → (T, H, W, 3) uint8
        pred_frames = ((decoded.cpu().float().clamp(-1, 1) + 1) / 2 * 255).to(_torch.uint8)
        pred_frames = pred_frames.permute(0, 2, 3, 1).numpy()

        # side-by-side: GT 위 / pred 아래
        T_out = min(len(pred_frames), len(primary_frames) - num_history)
        gt_vis   = primary_frames[num_history: num_history + T_out]
        pred_vis = pred_frames[:T_out]
        H, W, _ = gt_vis.shape[1:] if gt_vis.ndim == 4 else (gt_vis.shape[1], gt_vis.shape[2], 3)

        # resize pred to gt size if needed
        if pred_vis.shape[1:3] != gt_vis.shape[1:3]:
            resized = []
            for pf in pred_vis:
                resized.append(cv2.resize(pf, (W, H)))
            pred_vis = np.stack(resized)

        combined = np.concatenate([gt_vis, pred_vis], axis=1)  # (T, H*2, W, 3)
        video_path = os.path.join(exp_dir, 'rollout.mp4')
        mediapy.write_video(video_path, combined, fps=getattr(args, 'fps', 5))
        print(f"  saved: {video_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Gripper scale mismatch diagnostic')

    # ── 공통 ────────────────────────────────────────────────────────
    parser.add_argument('--episodes',       type=int, nargs='+', default=[85, 118],
                        help='진단할 에피소드 번호 목록')
    parser.add_argument('--val_dataset_dir', type=str,
                        default='/home/dgu/minyoung/droid_data/tracking',
                        help='droid_tracking 루트 (episode_XXXXXX/ 포함)')
    parser.add_argument('--subset_dir',     type=str, default=None,
                        help='droid_subset 루트 (비교용, optional)')
    parser.add_argument('--out_dir',        type=str, default='debug_gripper_scale',
                        help='출력 루트 디렉토리')
    parser.add_argument('--analyze_only',   action='store_true',
                        help='통계 분석만 수행 (WM forward 없음)')

    # ── rollout 실험 ─────────────────────────────────────────────────
    parser.add_argument('--run_rollout',    action='store_true',
                        help='WM forward 실험 실행')
    parser.add_argument('--modes',          type=str, nargs='+',
                        default=['original', 'force_close', 'rescale_06112'],
                        help='실험 모드')
    parser.add_argument('--start_idx',      type=int, default=0)
    parser.add_argument('--val_skip',       type=int, default=1,
                        help='annotation frame skip (15fps 데이터면 3, 5fps면 1)')
    parser.add_argument('--view_idx',       type=int, default=0)

    # ── 모델 경로 (rollout 시 필요) ──────────────────────────────────
    parser.add_argument('--ckpt_path',      type=str, default=None)
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--data_stat_path', type=str,
                        default='dataset_meta_info/droid/stat.json')

    args = parser.parse_args()

    # 절대경로 변환
    if not os.path.isabs(args.data_stat_path):
        args.data_stat_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.data_stat_path,
        )

    # ── 1. 분석 ──────────────────────────────────────────────────────
    analysis_dir = os.path.join(args.out_dir, 'analysis')
    analyze_episodes(
        args.val_dataset_dir,
        args.episodes,
        analysis_dir,
        subset_dir=args.subset_dir,
    )

    if args.analyze_only:
        print("\n[DONE] analyze_only mode — exiting without WM forward.")
        return

    if not args.run_rollout:
        print("\n[INFO] --run_rollout not specified. Use --run_rollout to run experiments.")
        return

    # ── 2. WM 실험 ───────────────────────────────────────────────────
    if not args.ckpt_path or not args.svd_model_path:
        print("[ERROR] --ckpt_path and --svd_model_path required for --run_rollout")
        return

    # config 로드
    from config import wm_args
    wm = wm_args()
    wm.val_model_path  = args.ckpt_path
    wm.ckpt_path       = args.ckpt_path
    wm.svd_model_path  = args.svd_model_path
    wm.data_stat_path  = args.data_stat_path
    wm.val_dataset_dir = args.val_dataset_dir
    # adapter 없이 실행
    if not hasattr(wm, 'adapter_ckpt'):
        wm.adapter_ckpt = None
    if not hasattr(wm, 'use_object_state'):
        wm.use_object_state = False
    if not hasattr(wm, 'use_warning'):
        wm.use_warning = False

    print("\n[INFO] Loading Ctrl-World (no pi0) ...")
    agent = build_agent(wm)

    for ep_id in args.episodes:
        run_experiments(
            agent        = agent,
            val_dataset_dir = args.val_dataset_dir,
            ep_id        = ep_id,
            start_idx    = args.start_idx,
            pred_step    = wm.pred_step,
            num_history  = wm.num_history,
            num_frames   = wm.num_frames,
            val_skip     = args.val_skip,
            out_root     = args.out_dir,
            modes        = args.modes,
            view_idx     = args.view_idx,
        )

    print(f"\n[DONE] All results → {args.out_dir}/")


if __name__ == '__main__':
    main()
