"""
rollout_interact_pi_online.py — Pi0 Policy + Object-Aware Online Generation Pipeline

파이프라인 흐름:
  [Initialization]
  1. get_traj_info()로 GT trajectory + video latents 로드
  2. SAM3Manager.initialize() → 첫 프레임에서 객체별 mask 탐지
  3. ObjectRegistry 초기화 (appearance=CLIP embedding, bbox, state)

  [Online Generation Loop]
  4. pi0 policy.infer() → cartesian_pose 예측
  5. forward_wm() — action conditioning으로 프레임 chunk 생성 (GT+pred side-by-side)
  6. 생성된 각 프레임에 대해:
     a. SAM3Manager.update_chunk() → mask 전파
     b. 3-tier recovery (soft / redetect / hard)
     c. ObjectRegistry 갱신 (interaction 판별, appearance)
  7. Negative detection → rollback+retry or 진행
  8. 저장: 메인 비디오 / SAM3 overlay / tracking log JSON / policy info JSON
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import datetime
import numpy as np
import cv2
import torch
import einops
import mediapy
import random
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation as R

from openpi.training import config as config_pi
from openpi.policies import policy_config
from openpi_client import image_tools

from accelerate import Accelerator
from argparse import ArgumentParser

from config import wm_args
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld, OBJ_INJECTION_SCALE, WARNING_INJECTION_SCALE
from models.utils import get_fk_solution
from models.object_registry import ObjectRegistry, FEATURE_DIM, MAX_OBJECTS, SHAPE_THRESH, AREA_THRESH, EXTENT_THRESH
from sam3_manager_new import SAM3ManagerNew as SAM3Manager


# ─────────────────────────────────────────────────────────────────
# 헬퍼 함수 (rollout_online.py에서 그대로)
# ─────────────────────────────────────────────────────────────────

def build_object_prompts_with_qwen(frame, instruction, qwen_model, qwen_processor, device):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    pil = Image.fromarray(frame)
    prompt_text = (
        f"Task: {instruction}\n"
        "List the main objects visible in this scene that are relevant to the task. "
        "Always include 'robot arm and end-effector'. "
        "Format: one object per line, no numbering, no extra text."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text",  "text":  prompt_text},
    ]}]
    text_input   = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text_input], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = qwen_model.generate(**inputs, max_new_tokens=128)
    response = qwen_processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    if not any('robot arm' in l.lower() for l in lines):
        lines = ['robot arm and end-effector'] + lines
    return lines


def is_interaction(robot_mask, obj_mask, iou_thresh=0.05):
    if robot_mask is None or obj_mask is None:
        return False
    intersection = (robot_mask & obj_mask).sum()
    union        = (robot_mask | obj_mask).sum()
    return (intersection / union) > iou_thresh if union > 0 else False


def update_registry(registry, sam_results, frame, robot_label,
                    clip_model, clip_processor, device,
                    iou_thresh=0.05, initial_areas=None):
    """SAM3 결과를 받아 ObjectRegistry 갱신 + interaction 판별."""
    print(f"[IOU THRESH] {iou_thresh}")
    robot_mask   = sam_results.get(robot_label, {}).get('mask')
    interact_info = {}

    for label, result in sam_results.items():
        mask   = result['mask']
        absent = result['absent']

        if absent or mask is None:
            registry.mark_absent(label)
            interact_info[label] = {"iou": None, "state": -1.0}
            continue

        appearance = (registry.extract_appearance(frame, mask, clip_model, clip_processor, device)
                      if clip_model is not None else np.zeros(512, dtype=np.float32))
        bbox = ObjectRegistry.mask_to_bbox(mask, frame.shape[:2])

        if label != robot_label and robot_mask is not None:
            intersection = float((robot_mask & mask).sum())
            init_area    = (initial_areas or {}).get(label, 0.0)
            if init_area > 0:
                metric = intersection / init_area
            else:
                union  = float((robot_mask | mask).sum())
                metric = intersection / union if union > 0 else 0.0
            state = 1.0 if metric > iou_thresh else 0.0
            print(f"[INTERACT] label={label} iou={metric:.4f} thresh={iou_thresh} state={state}")
        else:
            metric = None
            state  = 0.0

        shape_latent = ObjectRegistry.extract_shape_latent(mask)
        interact_info[label] = {"iou": round(metric, 6) if metric is not None else None, "state": state}
        registry.update(label=label, presence=1.0, appearance=appearance,
                         bbox=bbox, state=state, frame=frame, mask=mask,
                         shape_latent=shape_latent)
    return interact_info


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ─────────────────────────────────────────────────────────────────
# Agent 클래스 (rollout_interact_pi.py에서 그대로)
# ─────────────────────────────────────────────────────────────────

class agent():
    def __init__(self, args):
        args.val_model_path = args.ckpt_path
        self.args        = args
        self.accelerator = Accelerator()
        self.device      = self.accelerator.device
        self.dtype       = args.dtype

        # ── pi0 policy 로드 ──────────────────────────────────────
        if 'pi05' in args.policy_type:
            config = config_pi.get_config("pi05_droid")
        elif 'pi0fast' in args.policy_type:
            config = config_pi.get_config("pi0fast_droid")
        elif 'pi0' in args.policy_type:
            config = config_pi.get_config("pi0_droid")
        else:
            raise ValueError(f"Unknown policy type: {args.policy_type}")
        self.policy = policy_config.create_trained_policy(config, args.pi_ckpt, pytorch_device='cuda:1')

        # 바로 여기서 찍기
        for i in range(torch.cuda.device_count()):
            a = torch.cuda.memory_allocated(i) / 1024**3
            print(f"after pi0: GPU {i}: {a:.2f}GB")

        # ── Ctrl-World 모델 로드 ─────────────────────────────────
        self.model = CrtlWorld(args)
        missing, unexpected = self.model.load_state_dict(
            torch.load(args.val_model_path, map_location='cpu'), strict=False
        )
        _adapter_prefixes = ('shape_projector.', 'object_state_encoder.', 'warning_encoder.')
        missing_adapter  = [k for k in missing if any(k.startswith(p) for p in _adapter_prefixes)]
        missing_backbone = [k for k in missing if not any(k.startswith(p) for p in _adapter_prefixes)]
        if missing_adapter:
            print(f"[INFO] adapter keys not in base ckpt (will load from --adapter_ckpt): "
                  f"{set(k.split('.')[0] for k in missing_adapter)}")
        if missing_backbone:
            print(f"[WARN] unexpected missing backbone keys: {missing_backbone}")
        if unexpected:
            print(f"[WARN] unexpected keys: {unexpected}")
        # --adapter_ckpt 지정 시 adapter 가중치를 base 위에 올림
        if getattr(args, 'adapter_ckpt', None):
            adapter_state = torch.load(args.adapter_ckpt, map_location='cpu')
            m2, u2 = self.model.load_state_dict(adapter_state, strict=False)
            loaded_modules = set(k.split('.')[0] for k in adapter_state)
            print(f"[ADAPTER] loaded {len(adapter_state)} keys {loaded_modules} from {args.adapter_ckpt}")
            if u2:
                print(f"[ADAPTER] unexpected keys: {u2}")
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("load world model success")

        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat        = json.load(f)
            self.state_p01   = np.array(data_stat['state_01'])[None, :]
            self.state_p99   = np.array(data_stat['state_99'])[None, :]

        # ── action adapter (joint velocity → cartesian) ──────────
        if args.action_adapter is not None:
            from models.action_adapter.train2 import Dynamics
            self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.device)
            self.dynamics_model.load_state_dict(
                torch.load(args.action_adapter, map_location=self.device)
            )

    def normalize_bound(self, data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, id, start_idx=0, steps=8, skip=1):
        val_dataset_dir  = self.args.val_dataset_dir
        annotation_path  = f"{val_dataset_dir}/annotation/val/{id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno['action'])
            except:
                length = anno["video_length"]

        frames_ids = np.arange(start_idx, start_idx + steps * skip, skip)
        max_ids    = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)

        instruction = anno['texts'][0]
        car_action  = np.array(anno['states'])[frames_ids]
        joint_pos   = np.array(anno['joints'])[frames_ids]

        video_dict   = []
        video_latent = []
        for vid_id in range(len(anno['videos'])):
            video_path = f"{val_dataset_dir}/videos/val/{id}/{vid_id}.mp4"
            #/home/dgu/minyoung/Ctrl-World/dataset_example/droid_new_setup/videos/val/0001
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(range(length)).numpy()
            true_video = true_video[frames_ids]
            video_dict.append(true_video)

            # VAE 인코딩
            device = self.device
            tv = torch.from_numpy(true_video).to(self.dtype).to(device)
            x  = tv.permute(0, 3, 1, 2) / 255.0 * 2 - 1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                latents = []
                for i in range(0, len(x), 4): # 32 -> 4 프레임씩 VAE 인코딩 (메모리 절약)
                    latents.append(
                        vae.encode(x[i:i+4]).latent_dist.sample().mul_(vae.config.scaling_factor)
                    )
                x = torch.cat(latents, dim=0)
            video_latent.append(x)

        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond,
                   his_cond=None, text=None,
                   obj_state=None, obj_shape=None, warning_vec=None):
        """
        action_cond:       (num_history+pred_step, 7) np.ndarray
        video_latent_true: list of 3 tensors (pred_step, 4, h, w) — GT latents
        video_latent_cond: (1, 4, 72, 40) torch.Tensor
        his_cond:          (1, num_history, 4, 72, 40) torch.Tensor
        text:              str | None
        obj_state:         (MAX_OBJECTS, FEATURE_DIM) np.ndarray | None
        warning_vec:       (8,) np.ndarray | None
        obj_shape:         (MAX_OBJECTS, SHAPE_SIZE^2) np.ndarray | None

        returns:
          videos_cat:      (pred_step, H*2, W*3, 3) np.uint8 — GT 위 / 예측 아래 side-by-side
          true_video:      (3, pred_step, H, W, 3) np.uint8
          videos:          (3, pred_step, H, W, 3) np.uint8  ← video_dict_pred
          latents:         (3, pred_step, 4, h, w) torch.Tensor
        """
        args      = self.args
        pipeline  = self.model.pipeline
        image_cond = video_latent_cond

        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames + args.num_history, args.action_dim)

        # ── 예측 ────────────────────────────────────────────────
        with torch.no_grad():
            if text is not None:
                text_token = self.model.action_encoder(
                    action_cond, text, self.model.tokenizer, self.model.text_encoder
                )
            else:
                text_token = self.model.action_encoder(action_cond)

            # object token concat (use_object_state=True 일 때만)
            if self.model.use_object_state and self.model.object_state_encoder is not None:
                from models.object_registry import FEATURE_DIM, MAX_OBJECTS, SHAPE_SIZE
                _obj_state = (torch.tensor(obj_state, dtype=self.dtype, device=self.device).unsqueeze(0)
                              if obj_state is not None
                              else torch.zeros(1, MAX_OBJECTS, FEATURE_DIM, device=self.device, dtype=self.dtype))
                _obj_shape = (torch.tensor(obj_shape, dtype=self.dtype, device=self.device).unsqueeze(0)
                              if obj_shape is not None
                              else torch.zeros(1, MAX_OBJECTS, SHAPE_SIZE * SHAPE_SIZE, device=self.device, dtype=self.dtype))
                obj_shape_proj = self.model.shape_projector(_obj_shape)
                obj_feat       = torch.cat([_obj_state, obj_shape_proj], dim=-1)
                obj_tokens = self.model.object_state_encoder(obj_feat)   # (1, MAX_OBJECTS, 1024)

                # residual injection: 시퀀스 길이 유지, 기존 토큰에 perturbation으로 더함
                text_token = text_token.clone()
                n_obj = min(obj_tokens.shape[1], text_token.shape[1])
                _obj_scale_inj = getattr(args, 'obj_injection_scale', OBJ_INJECTION_SCALE)
                text_token[:, :n_obj] = text_token[:, :n_obj] + _obj_scale_inj * obj_tokens[:, :n_obj]

            # warning token residual (use_warning=True 일 때만)
            # text_token_base: obj-conditioned, no warning (standard CFG positive)
            # text_token_warn: obj + warning (negative guidance target)
            text_token_warn = None
            if self.model.use_warning and self.model.warning_encoder is not None:
                from models.warning_utils import WARNING_DIM
                _warning = (torch.tensor(warning_vec, dtype=self.dtype, device=self.device).unsqueeze(0)
                            if warning_vec is not None
                            else torch.zeros(1, WARNING_DIM, device=self.device, dtype=self.dtype))
                warning_token = self.model.warning_encoder(_warning)    # (1, 1, 1024)
                text_token_warn = text_token.clone()
                _warn_scale_inj = getattr(args, 'warning_injection_scale', WARNING_INJECTION_SCALE)
                text_token_warn[:, :1] = text_token_warn[:, :1] + _warn_scale_inj * warning_token

            # warning_guidance_scale=0 → warning_text not passed (no extra UNet pass)
            _warn_scale = getattr(args, 'warning_guidance_scale', 0.0)
            _warning_text = text_token_warn if (text_token_warn is not None and _warn_scale > 0.0) else None

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height * 3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
                warning_text=_warning_text,
                warning_guidance_scale=_warn_scale,
            )
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)
        # latents: (3, pred_step, 4, h, w)

        # ── GT 디코딩 ───────────────────────────────────────────
        true_video = torch.stack(video_latent_true, dim=0)  # (3, pred_step, 4, h, w)
        bsz, fnum  = true_video.shape[:2]
        decoded_gt = []
        tv_flat    = true_video.flatten(0, 1)
        for i in range(0, tv_flat.shape[0], args.decode_chunk_size):
            chunk = tv_flat[i:i+args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decoded_gt.append(pipeline.vae.decode(chunk, **{"num_frames": chunk.shape[0]}).sample)
        true_video = torch.cat(decoded_gt, dim=0).reshape(bsz, fnum, *decoded_gt[0].shape[1:])
        true_video = ((true_video / 2.0 + 0.5).clamp(0, 1) * 255)
        true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)

        # ── 예측 디코딩 ─────────────────────────────────────────
        bsz2, fnum2 = latents.shape[:2]
        decoded_pr  = []
        lat_flat    = latents.flatten(0, 1)
        for i in range(0, lat_flat.shape[0], args.decode_chunk_size):
            chunk = lat_flat[i:i+args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decoded_pr.append(pipeline.vae.decode(chunk, **{"num_frames": chunk.shape[0]}).sample)
        videos = torch.cat(decoded_pr, dim=0).reshape(bsz2, fnum2, *decoded_pr[0].shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)
        # videos: (3, pred_step, H, W, 3)

        # ── GT+예측 side-by-side concat (저장용) ─────────────────
        videos_cat = np.concatenate([true_video, videos], axis=-3)        # (3, pred_step, H*2, W, 3)
        videos_cat = np.concatenate([v for v in videos_cat], axis=-2).astype(np.uint8)
        # videos_cat: (pred_step, H*2, W*3, 3)

        return videos_cat, true_video, videos, latents

    def forward_policy(self, videos, state, joints, text, time_step=1):
        """
        pi0 policy inference → (policy_in_out, joint_pos_skip, state_fk_skip)
        """
        image1 = videos[1]
        image2 = videos[2]
        image1 = torch.from_numpy(image1).to(torch.uint8)
        image2 = torch.from_numpy(image2).to(torch.uint8)
        assert image1.shape == (192, 320, 3), f"image1 shape mismatch: {image1.shape}"

        image1 = torch.nn.functional.interpolate(
            image1.permute(2,0,1).unsqueeze(0).float(), size=(180, 320),
            mode='bilinear', align_corners=False
        ).squeeze(0).permute(1,2,0).to(torch.uint8).numpy()
        image2 = torch.nn.functional.interpolate(
            image2.permute(2,0,1).unsqueeze(0).float(), size=(180, 320),
            mode='bilinear', align_corners=False
        ).squeeze(0).permute(1,2,0).to(torch.uint8).numpy()

        example = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(image1, 224, 224),
            "observation/wrist_image_left":      image_tools.resize_with_pad(image2, 224, 224),
            "observation/joint_position":        joints[:7],
            "observation/gripper_position":      joints[-1:],
            "prompt":                            text,
        }
        action_chunk = self.policy.infer(example)["actions"]

        current_joint   = joints[None, :][:, :7]
        current_gripper = joints[None, :][:, 7:]
        idx = ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
               if 'pi05' in self.args.policy_type
               else [0,1,2,3,4,5,6,7,8,9,9,9,9,9,9])
        joint_vel   = action_chunk[:, :7][idx]
        gripper_pos = np.clip(action_chunk[:, 7:][idx], 0, self.args.gripper_max)

        joint_pos = self.dynamics_model(current_joint, joint_vel, None, training=False)
        state_fk  = []
        joint_pos   = np.concatenate([current_joint, joint_pos], axis=0)[:15]
        gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]
        for i in range(joint_pos.shape[0]):
            fk    = get_fk_solution(joint_pos[i, :7])
            r_rot = R.from_matrix(fk[:3, :3])
            state_fk.append(np.concatenate([fk[:3, 3], r_rot.as_euler('xyz'), gripper_pos[i]], axis=0))
        state_fk = np.array(state_fk)  # (15, 7)

        skip      = self.args.policy_skip_step
        valid_num = int(skip * (self.args.pred_step - 1))
        policy_in_out = {
            'joint_pos': joint_pos[:valid_num],
            'joint_vel': joint_vel[:valid_num],
            'state_fk':  state_fk[:valid_num],
        }
        state_fk_skip  = state_fk[::skip][:self.args.pred_step]       # (pred_step, 7)
        joint_pos_skip = joint_pos[::skip][:self.args.pred_step]       # (pred_step, 7)
        joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:, -1:]], axis=-1)  # (pred_step, 8)

        return policy_in_out, joint_pos_skip, state_fk_skip


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser()

    # ── Ctrl-World / pi 인자 ──────────────────────────────────────
    parser.add_argument('--svd_model_path',         type=str, default=None)
    parser.add_argument('--clip_model_path',        type=str, default=None)
    parser.add_argument('--ckpt_path',              type=str, default=None)
    parser.add_argument('--dataset_root_path',      type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names',          type=str, default=None)
    parser.add_argument('--task_type',              type=str, default=None)
    parser.add_argument('--pi_ckpt',                type=str, default=None,
                        help='pi0/pi05 체크포인트 경로 (필수)')

    # ── SAM3 / online 인자 ────────────────────────────────────────
    parser.add_argument('--sam3_ckpt',       type=str, default='/home/dgu/minyoung/sam3/checkpoints/sam3.pt')
    parser.add_argument('--qwen_model_path', type=str, default=None,
                        help='Qwen3-VL 모델 경로. None이면 object_labels 수동 지정')
    parser.add_argument('--object_labels',   type=str, default=None,
                        help='수동 지정. 예: "robot arm and end-effector,cup,pen"')
    parser.add_argument('--view_idx',        type=int,   default=1,
                        help='SAM3 tracking에 사용할 카메라 뷰 인덱스 (0~2)')
    parser.add_argument('--iou_interaction', type=float, default=0.05,
                        help='로봇팔-물체 interaction 판별 threshold')
    parser.add_argument('--rollback_on_neg',      action='store_true',
                        help='Negative event 발생 시 rollback+retry')
    parser.add_argument('--max_retries',          type=int, default=3)
    parser.add_argument('--rollback_trim_margin', type=int, default=1)
    parser.add_argument('--use_warning',          action='store_true',
                        help='warning conditioning 활성화 (WarningEncoder)')
    parser.add_argument('--use_obj_token',        action='store_true',
                        help='obj_state token UNet 주입 (fine-tune 완료 후)')
    parser.add_argument('--adapter_ckpt',         type=str, default=None,
                        help='adapter-only checkpoint 경로. base ckpt 위에 올림')
    parser.add_argument('--obj_scale',               type=float, default=OBJ_INJECTION_SCALE,
                        help='obj token residual injection scale (학습 시 사용한 값과 일치해야 함)')
    parser.add_argument('--warn_scale',              type=float, default=WARNING_INJECTION_SCALE,
                        help='warning token residual injection scale')
    parser.add_argument('--warning_guidance_scale', type=float, default=0.0,
                        help='CFG-style negative warning guidance α (0=off, e.g. 1.0~2.0)')
    parser.add_argument('--seed',                 type=int, default=42)
    parser.add_argument('--raw_only',             action='store_true',
                        help='SAM3 skip, plain Ctrl-World video만 저장')
    parser.add_argument('--upscale_scale', type=float, default=None,
                    help='upscale 배율. 예: 2.0 → 384×640 별도 영상 저장')
    parser.add_argument('--rollback_use_initial_real_frame', action='store_true',
    help='롤백 재초기화 시 generated last_good_frame 대신 최초 real frame 사용')

    args_new = parser.parse_args()

    # ── args 합성 (None이 아닌 값만 덮어쓰기) ─────────────────────
    args = wm_args(task_type=args_new.task_type)
    for k, v in vars(args_new).items():
        if v is not None:
            setattr(args, k, v)

    # CLI 플래그 → args 연결
    args.use_object_state         = args_new.use_obj_token
    args.use_warning              = args_new.use_warning
    args.adapter_ckpt             = args_new.adapter_ckpt
    args.warning_guidance_scale   = args_new.warning_guidance_scale
    args.obj_injection_scale      = args_new.obj_scale
    args.warning_injection_scale  = args_new.warn_scale

    print(f"[SEED] {args.seed}")
    set_seed(args.seed)

    def upscale_frames(frames: list, scale: float) -> list:
        H_orig, W_orig = frames[0].shape[:2]
        H_new, W_new   = int(H_orig * scale), int(W_orig * scale)
        result = []
        for f in frames:
            t = torch.from_numpy(f).permute(2,0,1).unsqueeze(0).float()
            t = torch.nn.functional.interpolate(
                t, size=(H_new, W_new), mode='bilinear', align_corners=False #bilinear bicubic
            )
            result.append(t.squeeze(0).permute(1,2,0).to(torch.uint8).numpy())
        return result

    # ── Agent (pi0 + Ctrl-World) ────────────────────────────────
    Agent    = agent(args)
    VIEW_IDX = args_new.view_idx

    # ── SAM3 ────────────────────────────────────────────────────
    sam_manager = SAM3Manager(checkpoint_path=args_new.sam3_ckpt, device="cuda:0")

    sam_manager_up = None
    if args_new.upscale_scale is not None:
        sam_manager_up = SAM3Manager(checkpoint_path=args_new.sam3_ckpt, device="cuda:0")

    # ── CLIP ────────────────────────────────────────────────────
    clip_model, clip_processor = None, None
    if args.clip_model_path and os.path.exists(str(args.clip_model_path)):
        from transformers import CLIPVisionModelWithProjection, CLIPProcessor
        print(f"Loading CLIP from {args.clip_model_path} ...")
        clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model_path).to(Agent.device)
        clip_model.eval()
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    else:
        print("[WARN] clip_model_path not found — appearance similarity disabled")

    # ── Qwen3-VL (optional) ──────────────────────────────────────
    qwen_model, qwen_processor = None, None
    if args_new.qwen_model_path and os.path.exists(args_new.qwen_model_path):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print(f"Loading Qwen3-VL from {args_new.qwen_model_path} ...")
        qwen_model = AutoModelForVision2Seq.from_pretrained(
            args_new.qwen_model_path, torch_dtype=torch.bfloat16
        ).to(Agent.device)
        qwen_processor = AutoProcessor.from_pretrained(args_new.qwen_model_path)

    interact_num = args.interact_num
    pred_step    = args.pred_step
    num_history  = args.num_history
    num_frames   = args.num_frames
    history_idx  = args.history_idx

    colors = [(0,255,0), (0,0,255), (0,150,150),
              (0,150,255), (0,255,150), (0,128,200), (0,128,0)]

    upscale_vis_to_save = []
    upscale_frames_to_save = []  # ← 이게 없음

    # ── trajectory 루프 ──────────────────────────────────────────
    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):

        # ── GT trajectory + video_latents 로드 ───────────────────
        eef_gt, joint_pos_gt, video_dict, video_latents, _ = Agent.get_traj_info(
            val_id_i, start_idx=start_idx_i, steps=int(pred_step * interact_num + 8)
        )
        print(f"text_i: {text_i}  eef[0]: {eef_gt[0]}  joint[0]: {joint_pos_gt[0]}")

        # ── SAM3 초기화 ──────────────────────────────────────────
        first_frame = video_dict[VIEW_IDX][0]  # (H, W, 3) uint8

        if qwen_model is not None:
            object_labels = build_object_prompts_with_qwen(
                first_frame, text_i, qwen_model, qwen_processor, str(Agent.device)
            )
        elif args_new.object_labels:
            object_labels = [l.strip() for l in args_new.object_labels.split(',')]
        else:
            object_labels = ['robot arm and end-effector']
        print(f"Object labels: {object_labels}")
        torch.cuda.empty_cache()

        # sam_manager.initialize 바로 위에 추가
        print("\n=== GPU Memory before SAM3 initialize ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved  = torch.cuda.memory_reserved(i) / 1024**3
            total     = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: allocated={allocated:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB")
        print("==========================================\n")


        sam_manager.initialize(first_frame, object_labels)

        #---------------------------------- chuga
        initial_anchor_frame = first_frame.copy()
        initial_anchor_box_r = {}

        for label in object_labels:
            init_mask = sam_manager.object_masks.get(label)
            if init_mask is None:
                initial_anchor_box_r[label] = None
                continue

            ys, xs = np.where(init_mask)
            if len(ys) == 0:
                initial_anchor_box_r[label] = None
                continue

            H, W = initial_anchor_frame.shape[:2]
            initial_anchor_box_r[label] = [
                float(xs.min() / W),
                float(ys.min() / H),
                float(xs.max() / W),
                float(ys.max() / H),
            ]
        #------------------------------------

        # 추가
        if sam_manager_up is not None:
            first_frame_up = upscale_frames([first_frame], args_new.upscale_scale)[0]
            sam_manager_up.initialize(first_frame_up, object_labels)

        # ── ObjectRegistry 초기화 ────────────────────────────────
        registry    = ObjectRegistry()
        robot_label = object_labels[0]
        for label in object_labels:
            registry.register(label)

        initial_appearances = {}
        for label in object_labels:
            mask = sam_manager.object_masks.get(label)
            if mask is not None:
                app = (registry.extract_appearance(first_frame, mask, clip_model, clip_processor, str(Agent.device))
                       if clip_model is not None else np.zeros(512, dtype=np.float32))
                initial_appearances[label] = app
                bbox         = ObjectRegistry.mask_to_bbox(mask, first_frame.shape[:2])
                shape_latent = ObjectRegistry.extract_shape_latent(mask)
                registry.update(label, presence=1.0, appearance=app, bbox=bbox, state=0.0,
                                 frame=first_frame, mask=mask, shape_latent=shape_latent)
            else:
                initial_appearances[label] = None
                registry.mark_absent(label)

        # ── history buffer 초기화 ────────────────────────────────
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1,4,72,40)
        assert first_latent.shape == (1, 4, 72, 40), f"first_latent shape: {first_latent.shape}"

        his_cond  = [first_latent]        * (num_history * 4)
        his_joint = [joint_pos_gt[0:1]]   * (num_history * 4)
        his_eef   = [eef_gt[0:1]]         * (num_history * 4)
        video_dict_pred = [v[0:1] for v in video_dict]  # 초기: GT 첫 프레임

        # ── 저장 버퍼 ────────────────────────────────────────────
        video_to_save      = []
        vis_frames_to_save = []
        vel_vis_frames_to_save = []
        info_to_save       = []
        tracking_log       = {label: [] for label in object_labels}
        frame_counter      = 0
        rollback_count     = 0
        false_gen_count    = 0

        uuid     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        text_id  = (text_i.replace(' ','_').replace(',','').replace('.','')
                         .replace("'","").replace('"',''))[:40]
        base_dir = f"{args.save_dir}/{args.task_name}/video"
        os.makedirs(base_dir, exist_ok=True)

        # ── SAM3 / rollback 상수 ─────────────────────────────────
        SAM3_CHUNK          = 3
        REDETECT_SIM_THRESH = 0.70
        TRIM_MARGIN         = args_new.rollback_trim_margin
        MAX_RETRIES         = args_new.max_retries
        CAUSE_SEVERITY      = {'occluded': 0.5, 'out_of_frame': 0.7, 'vanished': 1.0, 'crushed': 1.5}
        SOFT_RECOVER_MAX    = 3
        DEFAULT_STREAK_THRESH = 2
        DEFAULT_SCORE_THRESH  = 2.0
        ROBOT_STREAK_THRESH   = 4
        ROBOT_SCORE_THRESH    = 3.5

        bad_streak  = {label: 0   for label in object_labels}
        error_score = {label: 0.0 for label in object_labels}
        soft_streak = {label: 0   for label in object_labels}

        # ── Phase2 training data accumulators (all frames, no trim) ──
        phase2_log     = []   # list of per-frame dicts (WarningDataset format)
        phase2_latents = []   # list of (4, h, w) tensors for VIEW_IDX

        # ── while 기반 rollout loop ──────────────────────────────
        i                = 0
        step_retry_count = 0

        while i < interact_num:
            print(f"\n── Step {i+1}/{interact_num}  retry={step_retry_count} ──")

            # 스냅샷 (rollback 복원용)
            his_cond_snap        = list(his_cond)
            his_eef_snap         = list(his_eef)
            his_joint_snap       = list(his_joint)
            frame_counter_snap   = frame_counter
            registry_snap        = registry.snapshot()
            video_dict_pred_snap = [v.copy() for v in video_dict_pred]

            step_full     = []
            step_vis      = []
            step_vel_vis  = []
            step_log  = {label: [] for label in object_labels}

            # retry 시 에러 점수 감쇠
            if step_retry_count > 0:
                for label in object_labels:
                    bad_streak[label]  = int(bad_streak[label] * 0.5)
                    error_score[label] = round(error_score[label] * 0.5, 3)
            soft_streak = {label: 0 for label in object_labels}

            # GT video latents (forward_wm 비교용)
            start_id          = int(i * (pred_step - 1))
            end_id            = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            # ── Policy forward ────────────────────────────────
            print("### policy forward ###")
            current_joint = his_joint[-1][0]
            current_pose  = his_eef[-1][0]
            current_obs   = [v[-1] for v in video_dict_pred]
            policy_in_out, joint_pos, cartesian_pose = Agent.forward_policy(
                current_obs, current_pose, current_joint, text=text_i
            )
            print(f"cartesian_pose[0]={cartesian_pose[0]}  cartesian_pose[-1]={cartesian_pose[-1]}")

            # ── World Model forward ───────────────────────────
            print("### world model forward ###")
            print(f'task: {text_i}, traj_id: {val_id_i}, step: {i}/{interact_num}')
            action_cond    = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
            action_cond    = np.concatenate([action_cond, cartesian_pose], axis=0)
            his_latent     = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]

            obj_state_np = registry.to_padded_tensor(MAX_OBJECTS)        # (MAX_OBJECTS, FEATURE_DIM)
            obj_shape_np = registry.to_padded_shape_tensor(MAX_OBJECTS)  # (MAX_OBJECTS, SHAPE_SIZE^2)

            # warning_vec: inference 시점에는 현재 registry 상태로 soft 계산 (hard는 미래 미확인)
            warning_np = None
            if Agent.args.use_warning:
                from models.warning_utils import compute_warning_vec
                obj_records = []
                for lbl in registry.labels():
                    obj = registry.get(lbl)
                    obj_records.append({
                        'absent':        obj.presence < 0.5,
                        'cause':         None,      # inference 시 미래 cause 미확인
                        'bad_streak':    0,
                        'error_score':   0.0,
                        'iou':           None,
                        'state':         obj.state,
                        'shape_score':   obj.shape_score,
                        'shape_rejected': obj.shape_rejected,
                        'area_ratio':    obj.area_ratio,
                        'extent_ratio':  obj.extent_ratio,
                        'bbox':          obj.bbox.tolist(),
                    })
                warning_np = compute_warning_vec(obj_records)

            videos_cat, true_videos, video_dict_pred_new, predict_latents = Agent.forward_wm(
                action_cond, video_latent_true, current_latent,
                his_cond=his_latent,
                text=text_i if Agent.args.text_cond else None,
                obj_state=obj_state_np,
                obj_shape=obj_shape_np,
                warning_vec=warning_np,
            )
            # video_dict_pred_new: (3, pred_step, H, W, 3)
            # predict_latents:     (3, pred_step, 4, h, w)
            # videos_cat:          (pred_step, H*2, W*3, 3)

            # ── raw_only 모드 ─────────────────────────────────
            if args_new.raw_only:
                n_save = pred_step if i == interact_num - 1 else pred_step - 1
                video_to_save.extend([videos_cat[t] for t in range(n_save)])
                if args_new.upscale_scale is not None:
                    upscale_frames_to_save.extend(
                        upscale_frames([videos_cat[t] for t in range(n_save)], args_new.upscale_scale)
                    )
                his_joint.append(joint_pos[pred_step - 1][None, :])
                his_eef.append(cartesian_pose[pred_step - 1][None, :])
                his_cond.append(
                    torch.cat([predict_latents[v][pred_step - 1] for v in range(3)], dim=1).unsqueeze(0)
                )
                video_dict_pred = video_dict_pred_new
                info_to_save.append(policy_in_out)
                i += 1
                continue

            # SAM3 tracking용 프레임
            pred_frames_track = [video_dict_pred_new[VIEW_IDX][t] for t in range(pred_step)]
            pred_frames_full  = [videos_cat[t] for t in range(pred_step)]

            # ── SAM3 Chunk Tracking ───────────────────────────
            sam_results_flat = []
            for chunk_start in range(0, len(pred_frames_track), SAM3_CHUNK):
                chunk         = pred_frames_track[chunk_start: chunk_start + SAM3_CHUNK]
                chunk_results = sam_manager.update_chunk(chunk)
                sam_results_flat.extend(chunk_results)
                if any(
                    any(r.get("first_bad_t") is not None for r in fr.values())
                    for fr in chunk_results
                ):
                    break

            # upscale 병렬 tracking
            sam_results_flat_up = []
            if sam_manager_up is not None:
                pred_frames_track_up = upscale_frames(pred_frames_track, args_new.upscale_scale)
                for chunk_start in range(0, len(pred_frames_track_up), SAM3_CHUNK):
                    chunk_up      = pred_frames_track_up[chunk_start: chunk_start + SAM3_CHUNK]
                    chunk_results_up = sam_manager_up.update_chunk(chunk_up)
                    sam_results_flat_up.extend(chunk_results_up)

            # ── 프레임별 처리 ─────────────────────────────────
            neg_event_detected = False
            first_bad_t        = None

            for t, (frame_track, frame_full, sam_results) in enumerate(
                zip(pred_frames_track, pred_frames_full, sam_results_flat)
            ):
                redetected_labels     = set()
                soft_recovered_labels = set()
                recovery_tier         = {}
                rollback_candidate    = {}

                for label in object_labels:
                    result   = sam_results.get(label, {})
                    is_robot = (label == robot_label)

                    # robot arm: crushed 무시
                    if is_robot and result.get('cause') == 'crushed':
                        sam_results[label]['cause']  = None
                        sam_results[label]['absent'] = False
                        result = sam_results[label]

                    if not (result.get('absent') or result.get('mask') is None):
                        bad_streak[label]  = 0
                        soft_streak[label] = 0
                        error_score[label] = max(0.0, error_score[label] - 0.3)
                        recovery_tier[label]      = 0
                        rollback_candidate[label] = False
                        continue

                    cause = result.get('cause') or 'vanished'

                    # ── Tier 1: Soft recovery ─────────────────
                    if cause == 'occluded' and soft_streak[label] < SOFT_RECOVER_MAX:
                        obj       = registry.get(label)
                        # shape_score 낮으면 last_good_mask 신뢰도 우선
                        soft_mask = obj.last_good_mask.copy() if obj.last_good_mask is not None else None
                        if soft_mask is not None:
                            sam_results[label]['mask']   = soft_mask
                            sam_results[label]['absent'] = False
                            sam_results[label]['cause']  = None
                            soft_recovered_labels.add(label)
                        soft_streak[label] += 1
                        bad_streak[label]  += 1
                        error_score[label] += CAUSE_SEVERITY.get('occluded', 0.5)
                        recovery_tier[label] = 1
                        st = ROBOT_STREAK_THRESH if is_robot else DEFAULT_STREAK_THRESH
                        sc = ROBOT_SCORE_THRESH  if is_robot else DEFAULT_SCORE_THRESH
                        shape_bad = (not is_robot) and obj.shape_score < SHAPE_THRESH
                        rollback_candidate[label] = (bad_streak[label] >= st or error_score[label] >= sc
                                                     or shape_bad)
                        print(f"  [SOFT] '{label}': soft_streak={soft_streak[label]}/{SOFT_RECOVER_MAX}"
                              f"  streak={bad_streak[label]}  score={error_score[label]:.2f}"
                              f"  shape_score={obj.shape_score:.3f}  candidate={rollback_candidate[label]}")
                        continue

                    if cause == 'occluded':
                        cause = 'vanished'
                        sam_results[label]['cause'] = 'vanished'
                        print(f"  [ESCALATE] '{label}': soft_streak 초과 → vanished 승격")
                    soft_streak[label] = 0

                    # ── Tier 2: Re-detection ──────────────────
                    skip_redetect = (cause == 'crushed' and not is_robot)
                    if not skip_redetect:
                        print(f"  [REDETECT] '{label}': cause={cause}, re-detecting...")
                        redet_mask, _ = sam_manager.redetect(frame_track, label)
                        if redet_mask is not None:
                            resumed  = True
                            init_emb = initial_appearances.get(label)
                            if clip_model is not None and init_emb is not None and init_emb.sum() != 0:
                                curr_emb = registry.extract_appearance(
                                    frame_track, redet_mask, clip_model, clip_processor, str(Agent.device)
                                )
                                sim = float(np.dot(curr_emb, init_emb) /
                                            (np.linalg.norm(curr_emb) * np.linalg.norm(init_emb) + 1e-8))
                                print(f"  [REDETECT] '{label}': cosine_sim={sim:.4f}  thresh={REDETECT_SIM_THRESH}")
                                resumed = sim >= REDETECT_SIM_THRESH
                                if not resumed:
                                    sam_results[label]['cause'] = 'vanished'
                                    cause = 'vanished'
                            if resumed:
                                sam_results[label]['mask']   = redet_mask
                                sam_results[label]['absent'] = False
                                sam_results[label]['cause']  = None
                                redetected_labels.add(label)
                                bad_streak[label]  = 0
                                error_score[label] = max(0.0, error_score[label] - 0.5)
                                recovery_tier[label]      = 2
                                rollback_candidate[label] = False
                                print(f"  [REDETECT] '{label}': resumed  area={float(redet_mask.sum()):.0f}")
                                continue
                        else:
                            print(f"  [REDETECT] '{label}': not found → vanished")
                            sam_results[label]['cause'] = 'vanished'
                            cause = 'vanished'

                    # ── Tier 3: Hard failure ──────────────────
                    severity = CAUSE_SEVERITY.get(cause, 1.0)
                    bad_streak[label]  += 1
                    error_score[label] += severity
                    sam_results[label]['absent'] = True
                    recovery_tier[label] = 3
                    st = ROBOT_STREAK_THRESH if is_robot else DEFAULT_STREAK_THRESH
                    sc = ROBOT_SCORE_THRESH  if is_robot else DEFAULT_SCORE_THRESH
                    obj_hard      = registry.get(label)
                    shape_bad     = (not is_robot) and obj_hard.shape_score < SHAPE_THRESH
                    rollback_candidate[label] = (bad_streak[label] >= st or error_score[label] >= sc
                                                 or shape_bad)
                    print(f"  [SCORE] '{label}': cause={cause}"
                          f"  streak={bad_streak[label]}  score={error_score[label]:.2f}"
                          f"  shape_score={obj_hard.shape_score:.3f}  candidate={rollback_candidate[label]}")

                # rollback 판단 (non-robot 객체 기준)
                non_robot_trigger = [
                    lbl for lbl in object_labels
                    if lbl != robot_label and rollback_candidate.get(lbl, False)
                ]
                if non_robot_trigger and not neg_event_detected:
                    neg_event_detected = True
                    first_bad_t = min(max(0, t - bad_streak[lbl] + 1) for lbl in non_robot_trigger)
                    print(f"  [NEG] ROLLBACK triggered by {non_robot_trigger}  first_bad_t={first_bad_t}")

                # ObjectRegistry 갱신
                interact_info = update_registry(
                    registry, sam_results, frame_track, robot_label,
                    clip_model, clip_processor, str(Agent.device),
                    iou_thresh=args_new.iou_interaction,
                    initial_areas=sam_manager.initial_areas,
                )

                # tracking log
                for label in object_labels:
                    result = sam_results.get(label, {})
                    mask   = result.get('mask')
                    area   = float(mask.sum()) if mask is not None else 0.0
                    bbox   = None
                    if mask is not None:
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    H_f, W_f   = frame_track.shape[:2]
                    bbox_norm  = ([bbox[0]/W_f, bbox[1]/H_f, bbox[2]/W_f, bbox[3]/H_f]
                                  if bbox is not None else [0.0, 0.0, 0.0, 0.0])
                    shape_lat  = registry.get(label).to_shape_vector().tolist()
                    action_t   = (cartesian_pose[t].tolist()
                                  if t < len(cartesian_pose) else [0.0] * 7)
                    appearance_sim = None
                    if clip_model is not None and mask is not None:
                        curr_emb = registry.extract_appearance(
                            frame_track, mask, clip_model, clip_processor, str(Agent.device)
                        )
                        init_emb = initial_appearances.get(label)
                        if init_emb is not None and init_emb.sum() != 0:
                            sim = float(np.dot(curr_emb, init_emb) /
                                        (np.linalg.norm(curr_emb) * np.linalg.norm(init_emb) + 1e-8))
                            appearance_sim = round(sim, 4)
                            if result.get('cause') == 'crushed':
                                print(f"  [APPEARANCE] '{label}' crushed: cosine_sim={sim:.4f}")
                    info = interact_info.get(label, {})
                    step_log[label].append({
                        "frame":              frame_counter_snap + t,
                        "area":               area,
                        "bbox":               bbox,
                        "bbox_norm":          bbox_norm,
                        "shape_latent":       shape_lat,
                        "action":             action_t,
                        "absent":             result.get('absent', False),
                        "cause":              result.get('cause'),
                        "recovery_tier":      recovery_tier.get(label, 0),
                        "rollback_candidate": rollback_candidate.get(label, False),
                        "redetected":         label in redetected_labels,
                        "soft_recovered":     label in soft_recovered_labels,
                        "bad_streak":         bad_streak.get(label, 0),
                        "error_score":        round(error_score.get(label, 0.0), 3),
                        "iou":                info.get("iou"),
                        "state":              info.get("state"),
                        "appearance":         appearance_sim,
                        "shape_score":        round(registry.get(label).shape_score, 4),
                        "shape_rejected":     registry.get(label).shape_rejected,
                        "area_ratio":         round(registry.get(label).area_ratio, 4),
                        "extent_ratio":       round(registry.get(label).extent_ratio, 4),
                    })

                # Phase2: ALL 프레임 누적 (trim 없이, do_rollback 여부 무관)
                if not args_new.raw_only:
                    frame_obj = {}
                    for lbl in object_labels:
                        if step_log[lbl]:
                            e = step_log[lbl][-1]
                            frame_obj[lbl] = {
                                "absent":         e["absent"],
                                "cause":          e["cause"],
                                "bad_streak":     e["bad_streak"],
                                "error_score":    e["error_score"],
                                "iou":            e["iou"],
                                "state":          e["state"],
                                "shape_score":    e["shape_score"],
                                "shape_rejected": e["shape_rejected"],
                                "area_ratio":     e["area_ratio"],
                                "extent_ratio":   e["extent_ratio"],
                                "bbox":           e["bbox_norm"],
                                "appearance":     [],
                                "shape_latent":   e["shape_latent"],
                            }
                    phase2_log.append({
                        "frame_idx": frame_counter_snap + t,
                        "action":    action_t,
                        "objects":   frame_obj,
                    })
                    phase2_latents.append(predict_latents[VIEW_IDX][t].cpu())

                # SAM3 overlay
                vis_frame = frame_track.copy()
                for ci, (label, result) in enumerate(sam_results.items()):
                    mask = result.get('mask')
                    if mask is None:
                        print(f"  [MASK FAIL] '{label}' tracking lost at t={t}")
                        continue
                    color = colors[ci % len(colors)]
                    vis_frame[mask] = (vis_frame[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                    ys, xs = np.where(mask)
                    if len(ys) > 0:
                        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
                        vis_frame[y1:y1+2, x1:x2] = color
                        vis_frame[y2:y2+2, x1:x2] = color
                        vis_frame[y1:y2, x1:x1+2] = color
                        vis_frame[y1:y2, x2:x2+2] = color
                        final_absent = (result.get('absent') or mask is None
                                        or result.get('cause') in {'vanished', 'crushed'})
                        if final_absent:
                            vis_frame[y1:y2, x1:x2] = (
                                vis_frame[y1:y2, x1:x2] * 0.5 + np.array([255, 0, 0]) * 0.5
                            ).astype(np.uint8)

                step_full.append(frame_full)
                step_vis.append(vis_frame)

                # ── velocity 시각화 프레임 ────────────────────────────
                vel_frame = vis_frame.copy()
                H_vf, W_vf = vel_frame.shape[:2]
                for ci, (label, result) in enumerate(sam_results.items()):
                    mask = result.get('mask')
                    color_bgr = colors[ci % len(colors)]  # (R,G,B) → cv2는 BGR
                    cv2_color  = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))

                    # centroid 점
                    cent_hist = sam_manager._centroid_history.get(label, [])
                    valid_cents = [p for p in cent_hist if p is not None]
                    if valid_cents:
                        cx, cy = int(valid_cents[-1][0]), int(valid_cents[-1][1])
                        cv2.circle(vel_frame, (cx, cy), 5, (255, 255, 255), -1)
                        cv2.circle(vel_frame, (cx, cy), 5, cv2_color, 1)

                        # 속도 화살표 (최근 궤적 → 방향)
                        vel = sam_manager._centroid_velocity(label)
                        if vel is not None:
                            vx, vy = vel
                            scale = 4.0
                            ex = int(cx + vx * scale)
                            ey = int(cy + vy * scale)
                            ex = max(0, min(W_vf - 1, ex))
                            ey = max(0, min(H_vf - 1, ey))
                            cv2.arrowedLine(vel_frame, (cx, cy), (ex, ey),
                                            (255, 255, 0), 2, tipLength=0.4)

                        # 최근 centroid 궤적 (점선)
                        for pi in range(max(0, len(valid_cents)-5), len(valid_cents)-1):
                            p1 = (int(valid_cents[pi][0]),   int(valid_cents[pi][1]))
                            p2 = (int(valid_cents[pi+1][0]), int(valid_cents[pi+1][1]))
                            cv2.line(vel_frame, p1, p2, cv2_color, 1)

                    # border 접촉 엣지 강조 (노란색)
                    if mask is not None:
                        bm = 5
                        if mask[:bm, :].any():
                            cv2.line(vel_frame, (0, 0), (W_vf, 0), (0, 255, 255), 3)
                        if mask[-bm:, :].any():
                            cv2.line(vel_frame, (0, H_vf-1), (W_vf, H_vf-1), (0, 255, 255), 3)
                        if mask[:, :bm].any():
                            cv2.line(vel_frame, (0, 0), (0, H_vf), (0, 255, 255), 3)
                        if mask[:, -bm:].any():
                            cv2.line(vel_frame, (W_vf-1, 0), (W_vf-1, H_vf), (0, 255, 255), 3)

                    # cause 텍스트
                    cause = result.get('cause')
                    absent = result.get('absent', False)
                    if absent and cause:
                        cause_color = {
                            'occluded':     (255, 165,   0),
                            'out_of_frame': (  0, 255, 255),
                            'vanished':     (  0,   0, 255),
                            'crushed':      (255,   0, 255),
                        }.get(cause, (255, 255, 255))
                        cause_cv2 = (int(cause_color[2]), int(cause_color[1]), int(cause_color[0]))
                        if mask is not None:
                            ys_t, xs_t = np.where(mask)
                            tx = int(xs_t.min()) if len(ys_t) > 0 else 5
                            ty = int(ys_t.min()) - 6 if len(ys_t) > 0 else 15
                        else:
                            tx, ty = 5, 15 + ci * 15
                        ty = max(12, ty)
                        cv2.putText(vel_frame, cause, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cause_cv2, 1, cv2.LINE_AA)
                step_vel_vis.append(vel_frame)

                if sam_manager_up is not None and t < len(sam_results_flat_up):
                    sam_results_up = sam_results_flat_up[t]
                    frame_up       = pred_frames_track_up[t]
                    vis_frame_up   = frame_up.copy()
                    for ci, (label, result_up) in enumerate(sam_results_up.items()):
                        mask_up = result_up.get('mask')
                        if mask_up is None:
                            continue
                        color = colors[ci % len(colors)]
                        vis_frame_up[mask_up] = (vis_frame_up[mask_up] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                        ys, xs = np.where(mask_up)
                        if len(ys) > 0:
                            y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
                            vis_frame_up[y1:y1+2, x1:x2] = color
                            vis_frame_up[y2:y2+2, x1:x2] = color
                            vis_frame_up[y1:y2, x1:x1+2] = color
                            vis_frame_up[y1:y2, x2:x2+2] = color
                            final_absent_up = (result_up.get('absent') or
                               result_up.get('cause') in {'vanished', 'crushed'})
                            if final_absent_up:
                                vis_frame_up[y1:y2, x1:x2] = (
                                    vis_frame_up[y1:y2, x1:x2] * 0.5 + np.array([255, 0, 0]) * 0.5
                                ).astype(np.uint8)
                    upscale_vis_to_save.append(vis_frame_up)  # ← step 단위 임시 버퍼가 아닌 직접 누적

                if neg_event_detected:
                    break

            # ── 롤백 vs 진행 결정 ────────────────────────────────
            do_rollback = (neg_event_detected and args_new.rollback_on_neg
                           and step_retry_count < MAX_RETRIES)

            if do_rollback:
                trim_t = max(0, (first_bad_t or 0) - TRIM_MARGIN)
                print(f"[ROLLBACK] step={i}  retry={step_retry_count+1}/{MAX_RETRIES}"
                      f"  first_bad_t={first_bad_t}  trim_t={trim_t}")

                fg_frames = step_full[(first_bad_t if first_bad_t is not None else trim_t):]
                if fg_frames:
                    fg_arr  = np.stack(fg_frames, axis=0)
                    fg_path = (f"{base_dir}/false_gen_{uuid}_{val_id_i}_{start_idx_i}"
                               f"_step{i}_r{step_retry_count}.mp4")
                    mediapy.write_video(fg_path, fg_arr, fps=4)
                    print(f"[FALSE_GEN] {fg_path}  ({len(fg_frames)} frames)")
                    false_gen_count += 1

                video_to_save.extend(step_full[:trim_t])
                vis_frames_to_save.extend(step_vis[:trim_t])
                vel_vis_frames_to_save.extend(step_vel_vis[:trim_t])
                for label in object_labels:
                    tracking_log[label].extend(step_log[label][:trim_t])
                frame_counter = frame_counter_snap + trim_t

                # 복원
                his_cond        = list(his_cond_snap)
                his_eef         = list(his_eef_snap)
                his_joint       = list(his_joint_snap)
                video_dict_pred = video_dict_pred_snap
                registry.restore(registry_snap)

                #rollback code 추가
                #--rollback_use_initial_real_frame
                for label in object_labels:
                    obj = registry.get(label)
                    # shape_score / area_ratio / extent_ratio 기준으로 anchor 선택
                    use_initial = (
                        obj.shape_score   < SHAPE_THRESH  or
                        obj.area_ratio    < AREA_THRESH   or
                        obj.extent_ratio  < EXTENT_THRESH
                    )
                    if use_initial and obj.initial_good_frame is not None and obj.initial_good_mask is not None:
                        anchor_frame = obj.initial_good_frame
                        anchor_mask  = obj.initial_good_mask
                        print(f"  [ROLLBACK] '{label}': → initial anchor "
                              f"(shape={obj.shape_score:.2f} area={obj.area_ratio:.2f} ext={obj.extent_ratio:.2f})")
                    elif obj.last_good_frame is not None and obj.last_good_mask is not None:
                        anchor_frame = obj.last_good_frame
                        anchor_mask  = obj.last_good_mask
                        print(f"  [ROLLBACK] '{label}': → last_good anchor "
                              f"(shape={obj.shape_score:.2f} area={obj.area_ratio:.2f} ext={obj.extent_ratio:.2f})")
                    else:
                        continue
                    ys, xs = np.where(anchor_mask)
                    H, W   = anchor_frame.shape[:2]
                    if len(ys) > 0:
                        box_r = [float(xs.min()/W), float(ys.min()/H),
                                 float(xs.max()/W), float(ys.max()/H)]
                        sam_manager.set_anchor(label, anchor_frame, box_r)

                sam_manager.reset_session()


                '''for label in object_labels:
                    obj = registry.get(label)
                    if obj.last_good_frame is not None and obj.last_good_mask is not None:
                        ys, xs = np.where(obj.last_good_mask)
                        H, W   = obj.last_good_frame.shape[:2]
                        if len(ys) > 0:
                            box_r = [float(xs.min()/W), float(ys.min()/H),
                                     float(xs.max()/W), float(ys.max()/H)]
                            sam_manager.set_anchor(label, obj.last_good_frame, box_r)
                sam_manager.reset_session()'''

                if sam_manager_up is not None:
                    sam_manager_up.reset_session()

                step_retry_count += 1
                rollback_count   += 1
                continue

            # ── 진행 ─────────────────────────────────────────────
            step_retry_count = 0

            # rollback 없는 데이터 수집 모드: trim 없이 전 프레임 저장
            collect_mode = not args_new.rollback_on_neg
            if neg_event_detected and not collect_mode:
                trim_t = max(0, (first_bad_t or 0) - TRIM_MARGIN)
                n_save = trim_t
                print(f"[MAX_RETRIES] step={i}: saving {n_save} safe frames and continuing")
            elif i == interact_num - 1:
                n_save = len(step_full)
            else:
                n_save = min(pred_step - 1, len(step_full))

            video_to_save.extend(step_full[:n_save])
            if args_new.upscale_scale is not None:
                upscale_frames_to_save.extend(
                    upscale_frames(step_full[:n_save], args_new.upscale_scale)
                )
            vis_frames_to_save.extend(step_vis[:n_save])
            vel_vis_frames_to_save.extend(step_vel_vis[:n_save])
            for label in object_labels:
                tracking_log[label].extend(step_log[label][:n_save])
            frame_counter = frame_counter_snap + n_save

            # history 업데이트
            his_joint.append(joint_pos[pred_step - 1][None, :])
            his_eef.append(cartesian_pose[pred_step - 1][None, :])
            his_cond.append(
                torch.cat([predict_latents[v][pred_step - 1] for v in range(3)], dim=1).unsqueeze(0)
            )
            video_dict_pred = video_dict_pred_new
            info_to_save.append(policy_in_out)

            for label in object_labels:
                bad_streak[label]  = 0
                error_score[label] = 0.0
                soft_streak[label] = 0

            i += 1

        # ── 저장 ─────────────────────────────────────────────────
        print("##########################################################################")

        # 메인 비디오 (GT+pred side-by-side)
        filename_video = (
            f"{base_dir}/{args.task_type}_time_{uuid}_traj_{val_id_i}"
            f"_{start_idx_i}_{args.policy_skip_step}_{text_id}.mp4"
        )
        if args_new.upscale_scale is not None and upscale_frames_to_save:
            up_path = filename_video.replace('.mp4', f'_up{args_new.upscale_scale}x.mp4')
            mediapy.write_video(up_path, np.stack(upscale_frames_to_save, axis=0), fps=4)
            print(f"Upscaled video: {up_path}")

        if video_to_save:
            mediapy.write_video(filename_video, np.stack(video_to_save, axis=0), fps=4)
            print(f"Saved: {filename_video}  (rollbacks={rollback_count}, false_gens={false_gen_count})")

        # SAM3 overlay 영상
        if vis_frames_to_save:
            vis_path = filename_video.replace('.mp4', '_sam3.mp4')
            mediapy.write_video(vis_path, np.stack(vis_frames_to_save, axis=0), fps=4)
            print(f"SAM3 overlay: {vis_path}")

        # velocity 시각화 영상 (synthetic_traj/velocity/ 폴더)
        if vel_vis_frames_to_save:
            vel_dir = os.path.join(args.save_dir, "velocity")
            os.makedirs(vel_dir, exist_ok=True)
            vel_filename = os.path.basename(filename_video).replace('.mp4', '_vel.mp4')
            vel_path = os.path.join(vel_dir, vel_filename)
            mediapy.write_video(vel_path, np.stack(vel_vis_frames_to_save, axis=0), fps=4)
            print(f"Velocity vis: {vel_path}")

        if upscale_vis_to_save:
            up_vis_path = filename_video.replace('.mp4', f'_sam3_up{args_new.upscale_scale}x.mp4')
            mediapy.write_video(up_vis_path, np.stack(upscale_vis_to_save, axis=0), fps=4)
            print(f"SAM3 upscale overlay: {up_vis_path}")

        # 객체별 tracking log JSON
        for label, log in tracking_log.items():
            safe_label = label.replace(' ', '_').replace('/', '-')
            json_path  = filename_video.replace('.mp4', f'_{safe_label}.json')
            with open(json_path, 'w') as f:
                json.dump({
                    "label":        label,
                    "initial_area": sam_manager.initial_areas.get(label, 0.0),
                    "frames":       log,
                }, f, indent=2)
            print(f"Tracking log: {json_path}")

        # Phase2 training data (WarningDataset 포맷: tracking.json + latent.pt)
        if phase2_log and phase2_latents:
            phase2_dir = filename_video.replace('.mp4', '_phase2')
            os.makedirs(phase2_dir, exist_ok=True)
            phase2_meta = {
                "episode_id":           os.path.basename(phase2_dir),
                "object_labels":        object_labels,
                "language_instruction": text_i,
                "frames":               phase2_log,
            }
            with open(os.path.join(phase2_dir, 'tracking.json'), 'w') as f:
                json.dump(phase2_meta, f, indent=2)
            latent_tensor = torch.stack(phase2_latents, dim=0)  # (T, 4, h, w)
            torch.save(latent_tensor, os.path.join(phase2_dir, 'latent.pt'))
            print(f"Phase2 data: {phase2_dir}  ({len(phase2_log)} frames, "
                  f"latent={tuple(latent_tensor.shape)})")

        # policy info JSON
        if info_to_save:
            info     = {'success': 1, 'start_idx': 0,
                        'end_idx': len(video_to_save) - 1, 'instructions': text_i}
            for key in info_to_save[0].keys():
                info[key] = []
                for step_info in info_to_save:
                    info[key] += step_info[key].tolist()
            info_dir = f"{args.save_dir}/{args.task_name}/info"
            os.makedirs(info_dir, exist_ok=True)
            filename_info = (
                f"{info_dir}/{args.task_type}_time_{uuid}_traj_{val_id_i}"
                f"_{start_idx_i}_{pred_step}_{text_id}.json"
            )
            with open(filename_info, 'w') as f:
                json.dump(info, f, indent=4)
            print(f"Policy info: {filename_info}")

        print("##########################################################################")


# 실행 예:

'''CUDA_VISIBLE_DEVICES=0,1 python scripts/rollout_interact_pi_online.py \
  --task_type pickplace \
  --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt \
  --svd_model_path checkpoints/svd \
  --clip_model_path checkpoints/clip \
  --pi_ckpt /home/dgu/minyoung/checkpoints/pi05_droid_pytorch \
  --sam3_ckpt /home/dgu/minyoung/sam3/checkpoints/sam3.pt \
  --object_labels "robot arm and end-effector,pen" \
  --view_idx 1 \
  --seed 42 '''


'''python scripts/rollout_interact_pi_online.py \
  --task_type pickplace --seed 42 \
  --ckpt_path /path/to/finetuned.pt \
  --use_obj_token \
  --object_labels "pen"'''
