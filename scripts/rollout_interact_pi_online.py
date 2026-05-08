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

try:
    from openpi.training import config as config_pi
    from openpi.policies import policy_config
    from openpi_client import image_tools

    _OPENPI_AVAILABLE = True
except Exception:
    _OPENPI_AVAILABLE = False

from accelerate import Accelerator
from argparse import ArgumentParser

from config import wm_args
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld, OBJ_INJECTION_SCALE, WARNING_INJECTION_SCALE
from models.utils import get_fk_solution
from models.object_registry import (
    ObjectRegistry,
    FEATURE_DIM,
    MAX_OBJECTS,
    SHAPE_SIZE,
    SHAPE_THRESH,
    AREA_THRESH,
    EXTENT_THRESH,
)
from models.object_state_encoder import ObjectStateEncoder, N_OBJ, CROP_SZ
from sam3_manager_new import SAM3ManagerNew as SAM3Manager
from dataset.phase2_dataset_builder import Phase2DatasetBuilder

import tempfile
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# 헬퍼 함수 (rollout_online.py에서 그대로)
# ─────────────────────────────────────────────────────────────────


def build_object_prompts_with_qwen(
    frame, instruction, qwen_model, qwen_processor, device
):
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    pil = Image.fromarray(frame)
    prompt_text = (
        f"Task: {instruction}\n"
        "List the main objects visible in this scene that are relevant to the task. "
        "Always include 'robot arm and end-effector'. "
        "Format: one object per line, no numbering, no extra text."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text_input = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text_input], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = qwen_model.generate(**inputs, max_new_tokens=128)
    response = qwen_processor.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if not any("robot arm" in l.lower() for l in lines):
        lines = ["robot arm and end-effector"] + lines
    return lines


def is_interaction(robot_mask, obj_mask, iou_thresh=0.05):
    if robot_mask is None or obj_mask is None:
        return False
    intersection = (robot_mask & obj_mask).sum()
    union = (robot_mask | obj_mask).sum()
    return (intersection / union) > iou_thresh if union > 0 else False


def update_registry(
    registry,
    sam_results,
    frame,
    robot_label,
    clip_model,
    clip_processor,
    device,
    iou_thresh=0.05,
    initial_areas=None,
    detector_domain="original",
    scale_back_applied=False,
    scale_factor=1.0,
):
    """SAM3 결과를 받아 ObjectRegistry 갱신 + interaction 판별."""
    print(f"[IOU THRESH] {iou_thresh}")
    robot_mask = sam_results.get(robot_label, {}).get("mask")
    interact_info = {}

    for label, result in sam_results.items():
        mask = result["mask"]
        absent = result["absent"]

        if absent or mask is None:
            registry.mark_absent(label)
            interact_info[label] = {"iou": None, "state": -1.0}
            continue

        appearance = (
            registry.extract_appearance(frame, mask, clip_model, clip_processor, device)
            if clip_model is not None
            else np.zeros(512, dtype=np.float32)
        )
        bbox = ObjectRegistry.mask_to_bbox(mask, frame.shape[:2])

        if label != robot_label and robot_mask is not None:
            intersection = float((robot_mask & mask).sum())
            init_area = (initial_areas or {}).get(label, 0.0)
            if init_area > 0:
                metric = intersection / init_area
            else:
                union = float((robot_mask | mask).sum())
                metric = intersection / union if union > 0 else 0.0
            state = 1.0 if metric > iou_thresh else 0.0
            print(
                f"[INTERACT] label={label} iou={metric:.4f} thresh={iou_thresh} state={state}"
            )
        else:
            metric = None
            state = 0.0

        shape_latent = ObjectRegistry.extract_shape_latent(mask)
        interact_info[label] = {
            "iou": round(metric, 6) if metric is not None else None,
            "state": state,
        }
        registry.update(
            label=label,
            presence=1.0,
            appearance=appearance,
            bbox=bbox,
            state=state,
            frame=frame,
            mask=mask,
            shape_latent=shape_latent,
            detector_domain=detector_domain,
            scale_back_applied=scale_back_applied,
            scale_factor=scale_factor,
        )
    return interact_info


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def classify_failure_group(triggers, causes, registry):
    """failure_group 분류 (priority: presence > identity_shape > occlusion).
    returns (failure_group_str, representative_cause_str)
    """
    for lbl in triggers:
        c = causes.get(lbl)
        if c in {"vanished", "out_of_frame", "crushed"}:
            return "presence_recovery", c
    for lbl in triggers:
        obj = registry.get(lbl)
        if obj is not None:
            if (
                getattr(obj, "shape_rejected", False)
                or obj.shape_score < SHAPE_THRESH
                or obj.area_ratio < AREA_THRESH
            ):
                return "identity_shape_recovery", causes.get(lbl)
    for lbl in triggers:
        c = causes.get(lbl)
        if c == "occluded":
            return "occlusion_interaction_preserve", c
    return "presence_recovery", causes.get(triggers[0]) if triggers else None


# ─────────────────────────────────────────────────────────────────
# SAM3 Enhancement helpers
# ─────────────────────────────────────────────────────────────────


class SAM3Enhancer:
    """
    SAM3 입력 전용 프레임 업스케일러.
    반환된 mask/bbox는 반드시 original 해상도로 변환 후 registry에 넣어야 함.
    enhanced frame을 world model history/latent 조건에 절대 사용하지 말 것.
    """

    def __init__(
        self,
        mode="none",
        scale=4,
        realesrgan_root=None,
        realesrgan_model="RealESRGAN_x4plus",
        device="cuda:0",
    ):
        self.mode = mode
        self.scale = scale if mode != "none" else 1
        self._upsampler = None

        if mode == "realesrgan":
            if realesrgan_root is None:
                raise ValueError("--realesrgan_root required for realesrgan mode")
            sys.path.insert(0, realesrgan_root)
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            ckpt = os.path.join(realesrgan_root, "weights", f"{realesrgan_model}.pth")
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            self._upsampler = RealESRGANer(
                scale=scale,
                model_path=ckpt,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device=device,
            )
            print(f"[SAM3Enhancer] RealESRGAN x{scale} loaded from {ckpt}")
        elif mode == "opencv_sharpen":
            print(f"[SAM3Enhancer] opencv_sharpen x{scale}")
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown sam3_enhance_mode: {mode}")

    def enhance(self, frame):
        """frame: (H, W, 3) uint8 RGB → (H*scale, W*scale, 3) uint8 RGB"""
        if self.mode == "none":
            return frame
        if self.mode == "opencv_sharpen":
            H, W = frame.shape[:2]
            up = cv2.resize(
                frame, (W * self.scale, H * self.scale), interpolation=cv2.INTER_CUBIC
            )
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            return np.clip(cv2.filter2D(up, -1, kernel), 0, 255).astype(np.uint8)
        if self.mode == "realesrgan":
            bgr = frame[:, :, ::-1].copy()
            output, _ = self._upsampler.enhance(bgr, outscale=self.scale)
            return output[:, :, ::-1]  # BGR → RGB
        raise ValueError(f"Unknown mode: {self.mode}")

    def enhance_list(self, frames):
        return [self.enhance(f) for f in frames]

    def scale_points_up(self, points_dict):
        """point_coords를 enhanced 좌표계로 스케일 업."""
        if self.mode == "none" or points_dict is None:
            return points_dict
        scaled = {}
        for lbl, info in points_dict.items():
            pts = [
                [int(x * self.scale), int(y * self.scale)]
                for x, y in info["point_coords"]
            ]
            scaled[lbl] = {**info, "point_coords": pts}
        return scaled


def resize_mask_to_original(mask_enhanced, orig_H, orig_W):
    """bool mask (H_enh, W_enh) → bool mask (orig_H, orig_W) nearest-neighbor."""
    if mask_enhanced is None:
        return None
    m = mask_enhanced.astype(np.uint8) * 255
    return cv2.resize(m, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST) > 127


def sam3_result_to_original(
    result,
    orig_H,
    orig_W,
    detector_domain="original",
    scale_back_applied=False,
    scale_factor=1.0,
    downstream_domain="original",
):
    """SAM3 result dict의 mask를 original 해상도로 변환."""
    out = {k: v for k, v in result.items()}
    if result.get("mask") is not None:
        out["mask"] = resize_mask_to_original(result["mask"], orig_H, orig_W)

    out["detector_domain"] = detector_domain
    out["downstream_domain"] = downstream_domain
    out["scale_back_applied"] = bool(scale_back_applied)
    out["scale_factor"] = float(scale_factor)
    return out


def sam3_frame_results_to_original(
    frame_results,
    orig_H,
    orig_W,
    detector_domain="original",
    scale_back_applied=False,
    scale_factor=1.0,
    downstream_domain="original",
):
    """dict[label → result] enhanced space → original space."""
    return {
        lbl: sam3_result_to_original(
            r,
            orig_H,
            orig_W,
            detector_domain=detector_domain,
            scale_back_applied=scale_back_applied,
            scale_factor=scale_factor,
            downstream_domain=downstream_domain,
        )
        for lbl, r in frame_results.items()
    }


# ─────────────────────────────────────────────────────────────────
# Agent 클래스 (rollout_interact_pi.py에서 그대로)
# ─────────────────────────────────────────────────────────────────


class agent:
    def __init__(self, args):
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # ── pi0 policy 로드 (annotation_only 모드면 skip) ──────────
        if not getattr(args, "annotation_only", False):
            if not _OPENPI_AVAILABLE:
                raise RuntimeError(
                    "openpi is not installed. Use --annotation_only or install openpi."
                )
            if "pi05" in args.policy_type:
                config = config_pi.get_config("pi05_droid")
            elif "pi0fast" in args.policy_type:
                config = config_pi.get_config("pi0fast_droid")
            elif "pi0" in args.policy_type:
                config = config_pi.get_config("pi0_droid")
            else:
                raise ValueError(f"Unknown policy type: {args.policy_type}")
            self.policy = policy_config.create_trained_policy(
                config, args.pi_ckpt, pytorch_device="cuda:1"
            )
            for i in range(torch.cuda.device_count()):
                a = torch.cuda.memory_allocated(i) / 1024**3
                print(f"after pi0: GPU {i}: {a:.2f}GB")
        else:
            self.policy = None
            print("[annotation_only] pi0 policy 로드 skip")

        # ── Ctrl-World 모델 로드 ─────────────────────────────────
        self.model = CrtlWorld(args)
        missing, unexpected = self.model.load_state_dict(
            torch.load(args.val_model_path, map_location="cpu"), strict=False
        )
        _adapter_prefixes = (
            "shape_projector.",
            "object_state_encoder.",
            "warning_encoder.",
        )
        missing_adapter = [
            k for k in missing if any(k.startswith(p) for p in _adapter_prefixes)
        ]
        missing_backbone = [
            k for k in missing if not any(k.startswith(p) for p in _adapter_prefixes)
        ]
        if missing_adapter:
            print(
                f"[INFO] adapter keys not in base ckpt (will load from --adapter_ckpt): "
                f"{set(k.split('.')[0] for k in missing_adapter)}"
            )
        if missing_backbone:
            print(f"[WARN] unexpected missing backbone keys: {missing_backbone}")
        if unexpected:
            print(f"[WARN] unexpected keys: {unexpected}")
        # --adapter_ckpt 지정 시 Phase1 adapter 가중치를 로드
        # checkpoint 형식: {"obj_encoder": ..., "mid_block_obj_adapter": ...}
        self.obj_encoder = None
        if getattr(args, "adapter_ckpt", None):
            ckpt = torch.load(args.adapter_ckpt, map_location="cpu")
            # obj_encoder 로드
            self.obj_encoder = ObjectStateEncoder()
            if "obj_encoder" in ckpt:
                self.obj_encoder.load_state_dict(ckpt["obj_encoder"])
                print(f"[ADAPTER] obj_encoder loaded from {args.adapter_ckpt}")
            else:
                print(
                    f"[ADAPTER] WARNING: 'obj_encoder' key not found in {args.adapter_ckpt}"
                )
            # mid_block_obj_adapter 로드
            if "mid_block_obj_adapter" in ckpt:
                self.model.unet.mid_block_obj_adapter.load_state_dict(
                    ckpt["mid_block_obj_adapter"]
                )
                print(
                    f"[ADAPTER] mid_block_obj_adapter loaded from {args.adapter_ckpt}"
                )
            else:
                print(
                    f"[ADAPTER] WARNING: 'mid_block_obj_adapter' key not found in {args.adapter_ckpt}"
                )
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        if self.obj_encoder is not None:
            self.obj_encoder.to(self.accelerator.device).to(self.dtype)
            self.obj_encoder.eval()
        print("load world model success")

        with open(f"{args.data_stat_path}", "r") as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat["state_01"])[None, :]
            self.state_p99 = np.array(data_stat["state_99"])[None, :]

        # ── action adapter (joint velocity → cartesian) ──────────
        if args.action_adapter is not None:
            from models.action_adapter.train2 import Dynamics

            self.dynamics_model = Dynamics(
                action_dim=7, action_num=15, hidden_size=512
            ).to(self.device)
            self.dynamics_model.load_state_dict(
                torch.load(args.action_adapter, map_location=self.device)
            )

    def normalize_bound(
        self, data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8
    ):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, id, start_idx=0, steps=8, skip=1):
        val_dataset_dir = self.args.val_dataset_dir
        annotation_path = f"{val_dataset_dir}/annotation/val/{id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno["action"])
            except:
                length = anno["video_length"]

        frames_ids = np.arange(start_idx, start_idx + steps * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)

        instruction = anno["texts"][0]
        car_action = np.array(anno["states"])[frames_ids]
        joint_pos = np.array(anno["joints"])[frames_ids]

        video_dict = []
        video_latent = []
        for vid_id in range(len(anno["videos"])):
            video_path = f"{val_dataset_dir}/videos/val/{id}/{vid_id}.mp4"
            # /home/dgu/minyoung/Ctrl-World/dataset_example/droid_new_setup/videos/val/0001
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
            x = tv.permute(0, 3, 1, 2) / 255.0 * 2 - 1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                latents = []
                for i in range(
                    0, len(x), 4
                ):  # 32 -> 4 프레임씩 VAE 인코딩 (메모리 절약)
                    latents.append(
                        vae.encode(x[i : i + 4])
                        .latent_dist.sample()
                        .mul_(vae.config.scaling_factor)
                    )
                x = torch.cat(latents, dim=0)
            video_latent.append(x)

        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(
        self,
        action_cond,
        video_latent_true,
        video_latent_cond,
        his_cond=None,
        text=None,
        obj_state=None,
        obj_shape=None,
        warning_vec=None,
        condition_name=None,
        return_extra=False,
    ):
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
        args = self.args
        pipeline = self.model.pipeline
        image_cond = video_latent_cond

        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99)
        action_cond = (
            torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        )
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (
            args.num_frames + args.num_history,
            args.action_dim,
        )

        # ── 예측 ────────────────────────────────────────────────
        with torch.no_grad():
            extra = {
                "condition_name": condition_name,
                "obj_token_norm": None,
                "text_token_norm_before_obj": None,
                "text_token_norm_after_obj": None,
            }

            if text is not None:
                text_token = self.model.action_encoder(
                    action_cond, text, self.model.tokenizer, self.model.text_encoder
                )
            else:
                text_token = self.model.action_encoder(action_cond)

            extra["text_token_norm_before_obj"] = float(
                text_token.float().norm().item()
            )

            # 기존 residual injection (Phase1 adapter가 없을 때만 실행)
            # self.obj_encoder가 있으면 Phase1 cross-attn 경로를 사용하므로 이 블록은 skip
            if (
                self.model.use_object_state
                and self.model.object_state_encoder is not None
                and condition_name != "baseline_no_obj"
                and self.obj_encoder is None
            ):
                _obj_state = (
                    torch.tensor(
                        obj_state, dtype=self.dtype, device=self.device
                    ).unsqueeze(0)
                    if obj_state is not None
                    else torch.zeros(
                        1,
                        MAX_OBJECTS,
                        FEATURE_DIM,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                _obj_shape = (
                    torch.tensor(
                        obj_shape, dtype=self.dtype, device=self.device
                    ).unsqueeze(0)
                    if obj_shape is not None
                    else torch.zeros(
                        1,
                        MAX_OBJECTS,
                        SHAPE_SIZE * SHAPE_SIZE,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                obj_shape_proj = self.model.shape_projector(_obj_shape)
                obj_feat = torch.cat([_obj_state, obj_shape_proj], dim=-1)
                obj_tokens = self.model.object_state_encoder(
                    obj_feat
                )  # (1, MAX_OBJECTS, 1024)
                extra["obj_token_norm"] = float(obj_tokens.float().norm().item())

                # residual injection: 시퀀스 길이 유지, 기존 토큰에 perturbation으로 더함
                text_token = text_token.clone()
                n_obj = min(obj_tokens.shape[1], text_token.shape[1])
                _obj_scale_inj = getattr(
                    args, "obj_injection_scale", OBJ_INJECTION_SCALE
                )
                text_token[:, :n_obj] = (
                    text_token[:, :n_obj] + _obj_scale_inj * obj_tokens[:, :n_obj]
                )
                extra["text_token_norm_after_obj"] = float(
                    text_token.float().norm().item()
                )

            # warning token residual (use_warning=True 일 때만)
            # text_token_base: obj-conditioned, no warning (standard CFG positive)
            # text_token_warn: obj + warning (negative guidance target)
            text_token_warn = None
            if self.model.use_warning and self.model.warning_encoder is not None:
                from models.warning_utils import WARNING_DIM

                _warning = (
                    torch.tensor(
                        warning_vec, dtype=self.dtype, device=self.device
                    ).unsqueeze(0)
                    if warning_vec is not None
                    else torch.zeros(
                        1, WARNING_DIM, device=self.device, dtype=self.dtype
                    )
                )
                warning_token = self.model.warning_encoder(_warning)  # (1, 1, 1024)
                text_token_warn = text_token.clone()
                _warn_scale_inj = getattr(
                    args, "warning_injection_scale", WARNING_INJECTION_SCALE
                )
                text_token_warn[:, :1] = (
                    text_token_warn[:, :1] + _warn_scale_inj * warning_token
                )

            # warning_guidance_scale=0 → warning_text not passed (no extra UNet pass)
            _warn_scale = getattr(args, "warning_guidance_scale", 0.0)
            _warning_text = (
                text_token_warn
                if (text_token_warn is not None and _warn_scale > 0.0)
                else None
            )

            # ── Phase1 cross-attention adapter 경로 ─────────────
            # self.obj_encoder가 있으면 Phase1 adapter 사용
            # conditions:
            #   baseline_no_obj → None (adapter 완전 skip)
            #   adapter_zero    → zeros(1, F, N, 1024) (zero KV tokens)
            #   adapter_correct / adapter_shifted → ObjectStateEncoder로 인코딩
            _object_hidden_states = None
            if self.obj_encoder is not None:
                _F = args.num_history + args.num_frames
                if condition_name == "adapter_zero":
                    # adapter는 호출하되 zero KV로 — "content 없는 adapter ON" 베이스라인
                    _object_hidden_states = torch.zeros(
                        1, _F, N_OBJ, 1024, dtype=self.dtype, device=self.device
                    )
                    extra["obj_token_norm"] = 0.0
                elif condition_name != "baseline_no_obj" and obj_state is not None:
                    # obj_state: (N_OBJ, FEATURE_DIM), obj_shape: (N_OBJ, SHAPE_SIZE^2)
                    _N = N_OBJ
                    _os = (
                        torch.tensor(obj_state, dtype=self.dtype, device=self.device)
                        if not isinstance(obj_state, torch.Tensor)
                        else obj_state.to(device=self.device, dtype=self.dtype)
                    )
                    if _os.shape[0] != N_OBJ:
                        raise ValueError(
                            f"[phase1_eval] obj_state N mismatch: got {_os.shape[0]}, expected {N_OBJ}"
                        )
                    if _os.shape[-1] < 6:
                        raise ValueError(
                            f"[phase1_eval] obj_state FEATURE_DIM too small: {_os.shape}"
                        )
                    _osh = (
                        torch.tensor(obj_shape, dtype=self.dtype, device=self.device)
                        if obj_shape is not None
                        and not isinstance(obj_shape, torch.Tensor)
                        else (
                            obj_shape.to(device=self.device, dtype=self.dtype)
                            if obj_shape is not None
                            else torch.zeros(
                                _N,
                                SHAPE_SIZE * SHAPE_SIZE,
                                dtype=self.dtype,
                                device=self.device,
                            )
                        )
                    )
                    if _osh.shape[0] != N_OBJ:
                        raise ValueError(
                            f"[phase1_eval] obj_shape N mismatch: got {_osh.shape[0]}, expected {N_OBJ}"
                        )
                    # FEATURE_DIM layout: [presence(1), bbox(4), state(1)]
                    _pres = _os[:, 0:1]  # (N, 1)
                    _bbox = _os[:, 1:5]  # (N, 4)
                    _state_t = _os[:, 5:6]  # (N, 1)
                    _mask_crop = _osh.view(_N, CROP_SZ, CROP_SZ)  # (N, 16, 16)
                    # (1, F, N, ...) — 현재 상태를 모든 프레임에 반복
                    _pres_bf = (
                        _pres.unsqueeze(0)
                        .unsqueeze(0)
                        .expand(1, _F, -1, -1)
                        .contiguous()
                    )
                    _bbox_bf = (
                        _bbox.unsqueeze(0)
                        .unsqueeze(0)
                        .expand(1, _F, -1, -1)
                        .contiguous()
                    )
                    _st_bf = (
                        _state_t.unsqueeze(0)
                        .unsqueeze(0)
                        .expand(1, _F, -1, -1)
                        .contiguous()
                    )
                    _mk_bf = (
                        _mask_crop.unsqueeze(0)
                        .unsqueeze(0)
                        .expand(1, _F, -1, -1, -1)
                        .contiguous()
                    )
                    _object_hidden_states = self.obj_encoder(
                        _pres_bf, _bbox_bf, _st_bf, _mk_bf
                    )  # (1, F, N, 1024)
                    extra["obj_token_norm"] = float(
                        _object_hidden_states.float().norm().item()
                    )

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
                output_type="latent",
                return_dict=False,
                frame_level_cond=True,
                warning_text=_warning_text,
                warning_guidance_scale=_warn_scale,
                object_hidden_states=_object_hidden_states,
            )
        latents = einops.rearrange(
            latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1
        )
        # latents: (3, pred_step, 4, h, w)

        # ── GT 디코딩 ───────────────────────────────────────────
        true_video = torch.stack(video_latent_true, dim=0)  # (3, pred_step, 4, h, w)
        bsz, fnum = true_video.shape[:2]
        decoded_gt = []
        tv_flat = true_video.flatten(0, 1)
        for i in range(0, tv_flat.shape[0], args.decode_chunk_size):
            chunk = (
                tv_flat[i : i + args.decode_chunk_size]
                / pipeline.vae.config.scaling_factor
            )
            decoded_gt.append(
                pipeline.vae.decode(chunk, **{"num_frames": chunk.shape[0]}).sample
            )
        true_video = torch.cat(decoded_gt, dim=0).reshape(
            bsz, fnum, *decoded_gt[0].shape[1:]
        )
        true_video = (true_video / 2.0 + 0.5).clamp(0, 1) * 255
        true_video = (
            true_video.detach()
            .to(torch.float32)
            .cpu()
            .numpy()
            .transpose(0, 1, 3, 4, 2)
            .astype(np.uint8)
        )

        # ── 예측 디코딩 ─────────────────────────────────────────
        bsz2, fnum2 = latents.shape[:2]
        decoded_pr = []
        lat_flat = latents.flatten(0, 1)
        for i in range(0, lat_flat.shape[0], args.decode_chunk_size):
            chunk = (
                lat_flat[i : i + args.decode_chunk_size]
                / pipeline.vae.config.scaling_factor
            )
            decoded_pr.append(
                pipeline.vae.decode(chunk, **{"num_frames": chunk.shape[0]}).sample
            )
        videos = torch.cat(decoded_pr, dim=0).reshape(
            bsz2, fnum2, *decoded_pr[0].shape[1:]
        )
        videos = (videos / 2.0 + 0.5).clamp(0, 1) * 255
        videos = (
            videos.detach()
            .to(torch.float32)
            .cpu()
            .numpy()
            .transpose(0, 1, 3, 4, 2)
            .astype(np.uint8)
        )
        # videos: (3, pred_step, H, W, 3)

        # ── GT+예측 side-by-side concat (저장용) ─────────────────
        videos_cat = np.concatenate(
            [true_video, videos], axis=-3
        )  # (3, pred_step, H*2, W, 3)
        videos_cat = np.concatenate([v for v in videos_cat], axis=-2).astype(np.uint8)
        # videos_cat: (pred_step, H*2, W*3, 3)

        if return_extra:
            return videos_cat, true_video, videos, latents, extra
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

        image1 = (
            torch.nn.functional.interpolate(
                image1.permute(2, 0, 1).unsqueeze(0).float(),
                size=(180, 320),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .to(torch.uint8)
            .numpy()
        )
        image2 = (
            torch.nn.functional.interpolate(
                image2.permute(2, 0, 1).unsqueeze(0).float(),
                size=(180, 320),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .to(torch.uint8)
            .numpy()
        )

        example = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                image1, 224, 224
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(
                image2, 224, 224
            ),
            "observation/joint_position": joints[:7],
            "observation/gripper_position": joints[-1:],
            "prompt": text,
        }
        action_chunk = self.policy.infer(example)["actions"]

        current_joint = joints[None, :][:, :7]
        current_gripper = joints[None, :][:, 7:]
        idx = (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            if "pi05" in self.args.policy_type
            else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]
        )
        joint_vel = action_chunk[:, :7][idx]
        gripper_pos = np.clip(action_chunk[:, 7:][idx], 0, self.args.gripper_max)

        joint_pos = self.dynamics_model(current_joint, joint_vel, None, training=False)
        state_fk = []
        joint_pos = np.concatenate([current_joint, joint_pos], axis=0)[:15]
        gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]
        for i in range(joint_pos.shape[0]):
            fk = get_fk_solution(joint_pos[i, :7])
            r_rot = R.from_matrix(fk[:3, :3])
            state_fk.append(
                np.concatenate(
                    [fk[:3, 3], r_rot.as_euler("xyz"), gripper_pos[i]], axis=0
                )
            )
        state_fk = np.array(state_fk)  # (15, 7)

        skip = self.args.policy_skip_step
        valid_num = int(skip * (self.args.pred_step - 1))
        policy_in_out = {
            "joint_pos": joint_pos[:valid_num],
            "joint_vel": joint_vel[:valid_num],
            "state_fk": state_fk[:valid_num],
        }
        state_fk_skip = state_fk[::skip][: self.args.pred_step]  # (pred_step, 7)
        joint_pos_skip = joint_pos[::skip][: self.args.pred_step]  # (pred_step, 7)
        joint_pos_skip = np.concatenate(
            [joint_pos_skip, state_fk_skip[:, -1:]], axis=-1
        )  # (pred_step, 8)

        return policy_in_out, joint_pos_skip, state_fk_skip


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# Phase1 eval helpers
# ─────────────────────────────────────────────────────────────────


def build_phase1_condition(condition_name, registry, args_new, max_objects):
    """
    condition별 (obj_state_np, obj_shape_np) 반환.
    FEATURE_DIM layout: [presence(1), bbox(4), state(1)] — bbox indices 1:5
    """
    from models.object_registry import FEATURE_DIM

    if condition_name == "baseline_no_obj":
        return None, None  # forward_wm에서 injection 자체를 skip

    obj_state_np = registry.to_padded_tensor(max_objects)  # (max_objects, FEATURE_DIM)
    obj_shape_np = registry.to_padded_shape_tensor(
        max_objects
    )  # (max_objects, SHAPE_SIZE^2)

    # Debug/assert FEATURE_DIM layout: [presence, x1, y1, x2, y2, state, ...]
    _pres = obj_state_np[:, 0]
    _bbox = obj_state_np[:, 1:5]

    if not np.all((_pres >= -1e-4) & (_pres <= 1.0 + 1e-4)):
        raise ValueError(f"[phase1_eval] invalid presence values: {_pres}")

    _present = _pres >= 0.5
    if _present.any():
        _pb = _bbox[_present]
        if not np.all((_pb >= -1e-4) & (_pb <= 1.0 + 1e-4)):
            raise ValueError(f"[phase1_eval] bbox not normalized [0,1]: {_pb}")
        if not np.all(_pb[:, 2] > _pb[:, 0]):
            raise ValueError(f"[phase1_eval] invalid bbox x2<=x1: {_pb}")
        if not np.all(_pb[:, 3] > _pb[:, 1]):
            raise ValueError(f"[phase1_eval] invalid bbox y2<=y1: {_pb}")

    if condition_name == "adapter_zero":
        return np.zeros_like(obj_state_np), np.zeros_like(obj_shape_np)

    if condition_name == "adapter_correct":
        return obj_state_np, obj_shape_np

    if condition_name == "adapter_shifted":
        shifted = obj_state_np.copy()
        sx, sy = args_new.phase1_shift_x, args_new.phase1_shift_y
        for idx in range(max_objects):
            if shifted[idx, 0] < 0.5:  # absence → skip
                continue
            x1, y1, x2, y2 = (
                shifted[idx, 1],
                shifted[idx, 2],
                shifted[idx, 3],
                shifted[idx, 4],
            )
            w = x2 - x1
            h_ = y2 - y1
            cx = (x1 + x2) / 2 + sx
            cy = (y1 + y2) / 2 + sy
            cx = float(np.clip(cx, w / 2, 1.0 - w / 2))
            cy = float(np.clip(cy, h_ / 2, 1.0 - h_ / 2))
            shifted[idx, 1] = cx - w / 2
            shifted[idx, 3] = cx + w / 2
            shifted[idx, 2] = cy - h_ / 2
            shifted[idx, 4] = cy + h_ / 2
        return shifted, obj_shape_np

    raise ValueError(f"Unknown phase1 condition: {condition_name}")


def _pack_views_horiz(videos):
    """videos: (3, T, H, W, 3) uint8 → (T, H, W*3, 3)"""
    return np.concatenate([videos[v] for v in range(3)], axis=2)


def _save_diff_video(a, b, path, fps=4, amplify=5.0):
    """a, b: (T, H, W, 3) uint8 → diff mp4"""
    diff = (
        (np.abs(a.astype(np.float32) - b.astype(np.float32)) * amplify)
        .clip(0, 255)
        .astype(np.uint8)
    )
    mediapy.write_video(path, diff, fps=fps)


def _lat_mse(a, b):
    """a, b: torch.Tensor → float"""
    return float(((a.float() - b.float()) ** 2).mean().item())


def _px_mse(a, b):
    """a, b: np.ndarray uint8 → float"""
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _region_metric_single_view(vid_a, vid_b, bbox_norm, presence, view_H, view_W):
    """
    VIEW_IDX 기준 single view에서만 object/background region MSE 계산.
    vid_a, vid_b: (T, H, W, 3) uint8  ← VIEW_IDX frames만
    bbox_norm:    (MAX_OBJECTS, 4) normalized [x1,y1,x2,y2]
    presence:     (MAX_OBJECTS,) float
    """
    mask = np.zeros((view_H, view_W), dtype=bool)
    for n in range(len(presence)):
        if presence[n] < 0.5:
            continue
        x1, y1, x2, y2 = bbox_norm[n]
        px1 = int(np.clip(x1 * view_W, 0, view_W - 1))
        px2 = int(np.clip(x2 * view_W, 0, view_W))
        py1 = int(np.clip(y1 * view_H, 0, view_H - 1))
        py2 = int(np.clip(y2 * view_H, 0, view_H))
        mask[py1:py2, px1:px2] = True

    T = vid_a.shape[0]
    obj_mse = (
        float(
            np.mean(
                [
                    np.mean(
                        (
                            vid_a[t][mask].astype(np.float32)
                            - vid_b[t][mask].astype(np.float32)
                        )
                        ** 2
                    )
                    for t in range(T)
                ]
            )
        )
        if mask.any()
        else 0.0
    )

    bg_mse = (
        float(
            np.mean(
                [
                    np.mean(
                        (
                            vid_a[t][~mask].astype(np.float32)
                            - vid_b[t][~mask].astype(np.float32)
                        )
                        ** 2
                    )
                    for t in range(T)
                ]
            )
        )
        if (~mask).any()
        else 0.0
    )

    return {"obj_mse": obj_mse, "bg_mse": bg_mse, "ratio": obj_mse / (bg_mse + 1e-8)}


# -------------------------------------------------------------------------------------
def _bbox_stats_from_xyxy(bbox, init_bbox_area=None, prev_bbox=None):
    """
    bbox: [x1, y1, x2, y2] pixel coords or None
    init_bbox_area: initial bbox area in original pixel coords
    prev_bbox: previous bbox [x1, y1, x2, y2] or None

    returns JSON-safe bbox audit stats.
    """
    if bbox is None:
        return {
            "bbox_area": 0.0,
            "bbox_area_ratio": 0.0,
            "bbox_aspect": 0.0,
            "bbox_center": None,
            "bbox_jump_px": None,
        }

    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    aspect = w / (h + 1e-8)
    center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    if init_bbox_area is not None and init_bbox_area > 0:
        area_ratio = area / float(init_bbox_area)
    else:
        area_ratio = None

    if prev_bbox is not None:
        px1, py1, px2, py2 = [float(v) for v in prev_bbox]
        prev_center = [(px1 + px2) / 2.0, (py1 + py2) / 2.0]
        jump = float(
            np.sqrt(
                (center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2
            )
        )
    else:
        jump = None

    return {
        "bbox_area": round(float(area), 4),
        "bbox_area_ratio": (
            round(float(area_ratio), 4) if area_ratio is not None else None
        ),
        "bbox_aspect": round(float(aspect), 4),
        "bbox_center": [round(float(center[0]), 4), round(float(center[1]), 4)],
        "bbox_jump_px": round(float(jump), 4) if jump is not None else None,
    }


def _is_audit_good_frame(entry):
    """
    This is audit-only.
    It does not control actual registry update.
    Used only to record last_good/update_rejected metadata.
    """
    if entry is None:
        return False
    if entry.get("absent", False):
        return False
    if entry.get("rollback_candidate", False):
        return False
    if entry.get("shape_rejected", False):
        return False
    if (
        entry.get("area_ratio") is not None
        and entry.get("area_ratio", 0.0) < AREA_THRESH
    ):
        return False
    if (
        entry.get("shape_score") is not None
        and entry.get("shape_score", 0.0) < SHAPE_THRESH
    ):
        return False
    if (
        entry.get("extent_ratio") is not None
        and entry.get("extent_ratio", 0.0) < EXTENT_THRESH
    ):
        return False
    return True


def _find_last_good_entry(entries):
    """
    Search previous logged entries only.
    Do not include the current frame before append.
    """
    for e in reversed(entries):
        if _is_audit_good_frame(e):
            return e
    return None


def _audit_update_reject_reason(entry):
    reasons = []

    if entry.get("absent", False):
        reasons.append("absent")
    if entry.get("rollback_candidate", False):
        reasons.append("rollback_candidate")
    if entry.get("shape_rejected", False):
        reasons.append("shape_rejected")
    if (
        entry.get("area_ratio") is not None
        and entry.get("area_ratio", 1.0) < AREA_THRESH
    ):
        reasons.append("area_ratio_low")
    if (
        entry.get("shape_score") is not None
        and entry.get("shape_score", 1.0) < SHAPE_THRESH
    ):
        reasons.append("shape_score_low")
    if (
        entry.get("extent_ratio") is not None
        and entry.get("extent_ratio", 1.0) < EXTENT_THRESH
    ):
        reasons.append("extent_ratio_low")

    return "_".join(reasons) if reasons else None


def _tracking_status_from_entry(entry):
    """
    Human audit helper only.
    Do not use this as final automatic Phase2 label.
    """
    if entry.get("absent", False):
        return "tracking_lost"
    if entry.get("bbox") is None:
        return "tracking_lost"
    if entry.get("shape_rejected", False):
        return "tracking_suspicious"
    if (
        entry.get("area_ratio") is not None
        and entry.get("area_ratio", 1.0) < AREA_THRESH
    ):
        return "tracking_suspicious"
    if (
        entry.get("shape_score") is not None
        and entry.get("shape_score", 1.0) < SHAPE_THRESH
    ):
        return "tracking_suspicious"
    if (
        entry.get("extent_ratio") is not None
        and entry.get("extent_ratio", 1.0) < EXTENT_THRESH
    ):
        return "tracking_suspicious"
    return "tracking_ok"


# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()

    # ── Ctrl-World / pi 인자 ──────────────────────────────────────
    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument("--task_type", type=str, default=None)
    parser.add_argument(
        "--pi_ckpt", type=str, default=None, help="pi0/pi05 체크포인트 경로 (필수)"
    )

    # ── SAM3 / online 인자 ────────────────────────────────────────
    parser.add_argument(
        "--sam3_ckpt", type=str, default="/home/dgu/minyoung/sam3/checkpoints/sam3.pt"
    )
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default=None,
        help="Qwen3-VL 모델 경로. None이면 object_labels 수동 지정",
    )
    parser.add_argument(
        "--object_labels",
        type=str,
        default=None,
        help='수동 지정. 예: "robot arm and end-effector,cup,pen"',
    )
    parser.add_argument(
        "--view_idx",
        type=int,
        default=1,
        help="SAM3 tracking에 사용할 카메라 뷰 인덱스 (0~2)",
    )
    parser.add_argument(
        "--iou_interaction",
        type=float,
        default=0.05,
        help="로봇팔-물체 interaction 판별 threshold",
    )
    parser.add_argument(
        "--rollback_on_neg",
        action="store_true",
        help="Negative event 발생 시 rollback+retry",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--rollback_trim_margin", type=int, default=1)
    parser.add_argument(
        "--use_warning",
        action="store_true",
        help="warning conditioning 활성화 (WarningEncoder)",
    )
    parser.add_argument(
        "--use_obj_token",
        action="store_true",
        help="obj_state token UNet 주입 (fine-tune 완료 후)",
    )
    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default=None,
        help="adapter-only checkpoint 경로. base ckpt 위에 올림",
    )
    parser.add_argument(
        "--obj_scale",
        type=float,
        default=OBJ_INJECTION_SCALE,
        help="obj token residual injection scale (학습 시 사용한 값과 일치해야 함)",
    )
    parser.add_argument(
        "--warn_scale",
        type=float,
        default=WARNING_INJECTION_SCALE,
        help="warning token residual injection scale",
    )
    parser.add_argument(
        "--warning_guidance_scale",
        type=float,
        default=0.0,
        help="CFG-style negative warning guidance α (0=off, e.g. 1.0~2.0)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sam3_init_fallback_sr",
        action="store_true",
        help="Initial frame에서 original 먼저 detect하고 실패 label만 SR/enhanced에서 fallback detect",
    )
    # ── Phase1 eval mode ─────────────────────────────────────────
    parser.add_argument(
        "--phase1_eval_mode",
        action="store_true",
        help="rollback/SAM3 없이 조건별 generation 차이만 평가",
    )
    parser.add_argument(
        "--phase1_eval_out", type=str, default="eval_results/rollout_phase1_eval"
    )
    parser.add_argument(
        "--phase1_eval_steps", type=int, default=1, help="평가할 rollout step 수"
    )
    parser.add_argument(
        "--phase1_eval_conditions",
        nargs="+",
        default=[
            "baseline_no_obj",
            "adapter_zero",
            "adapter_correct",
            "adapter_shifted",
        ],
    )
    parser.add_argument(
        "--phase1_shift_x",
        type=float,
        default=0.2,
        help="adapter_shifted condition bbox center shift (x)",
    )
    parser.add_argument(
        "--phase1_shift_y",
        type=float,
        default=0.15,
        help="adapter_shifted condition bbox center shift (y)",
    )
    parser.add_argument(
        "--phase2_data_dir",
        type=str,
        default=None,
        help="Phase2 학습 샘플 저장 경로 (failure/risk 발생 시 저장)",
    )
    parser.add_argument(
        "--raw_only",
        action="store_true",
        help="SAM3 skip, plain Ctrl-World video만 저장",
    )
    parser.add_argument(
        "--upscale_scale",
        type=float,
        default=None,
        help="upscale 배율. 예: 2.0 → 384×640 별도 영상 저장",
    )
    parser.add_argument(
        "--rollback_use_initial_real_frame",
        action="store_true",
        help="롤백 재초기화 시 generated last_good_frame 대신 최초 real frame 사용",
    )
    parser.add_argument(
        "--debug_first_frame",
        action="store_true",
        help="첫 프레임 + SAM3 마스크 저장 후 즉시 종료 (디버그용)",
    )
    parser.add_argument(
        "--init_masks_dir",
        type=str,
        default=None,
        help="수동 마스크 PNG 디렉토리. 0.png=labels[0], 1.png=labels[1] ... "
        "SAM3 text 탐지 대신 해당 마스크 기반 point prompt 사용",
    )
    # ── SAM3 Enhancement ─────────────────────────────────────────
    parser.add_argument(
        "--sam3_enhance_mode",
        type=str,
        default="none",
        choices=["none", "realesrgan", "opencv_sharpen"],
        help="SAM3 입력 프레임 화질 개선 방법 (none=비활성)",
    )
    parser.add_argument(
        "--sam3_enhance_scale", type=int, default=4, help="화질 개선 배율 (기본 4)"
    )
    parser.add_argument(
        "--realesrgan_root",
        type=str,
        default="/home/dgu/minyoung/Real-ESRGAN",
        help="Real-ESRGAN 루트 디렉토리",
    )
    parser.add_argument(
        "--realesrgan_model",
        type=str,
        default="RealESRGAN_x4plus",
        help="Real-ESRGAN 모델명 (weights/ 아래 .pth 파일명)",
    )
    parser.add_argument(
        "--sam3_enhance_debug",
        action="store_true",
        help="original/enhanced/mask overlay PNG 저장 (디버그용)",
    )
    parser.add_argument(
        "--sam3_enhance_cache_dir",
        type=str,
        default="eval_results/sam3_enhance_cache",
        help="enhance debug PNG 저장 경로",
    )
    # ── trajectory 직접 지정 (config.py 설정 덮어쓰기) ───────────
    parser.add_argument(
        "--val_id",
        type=str,
        nargs="+",
        default=None,
        help="trajectory ID 직접 지정. 예: --val_id 28  "
        "(미지정 시 config.py 값 사용)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        nargs="+",
        default=None,
        help="시작 프레임 인덱스 직접 지정. 예: --start_idx 24  "
        "(미지정 시 config.py 값 사용)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        nargs="+",
        default=None,
        help='task instruction 직접 지정. 예: --instruction "pick the pen"  '
        "(미지정 시 config.py 값 사용)",
    )
    parser.add_argument(
        "--val_skip",
        type=int,
        default=None,
        help="get_traj_info frame skip. 원본 15fps 데이터는 3, "
        "이미 5fps로 처리된 데이터는 1 (미지정 시 config.down_sample 사용)",
    )
    parser.add_argument(
        "--interact_num",
        type=int,
        default=None,
        help="WM forward 반복 횟수 (미지정 시 config 값 사용). "
        "높을수록 더 긴 영상 생성 — hallucination 유도 시 30~50 권장",
    )
    parser.add_argument(
        "--annotation_only",
        action="store_true",
        help="pi0 정책 없이 annotation GT eef를 action condition으로 직접 사용",
    )

    args_new = parser.parse_args()

    # ── args 합성 (None이 아닌 값만 덮어쓰기) ─────────────────────
    args = wm_args(task_type=args_new.task_type)
    for k, v in vars(args_new).items():
        if v is not None:
            setattr(args, k, v)
    # val_id / start_idx / instruction CLI 오버라이드
    # 위의 일반 merge로도 처리되지만 명시적으로 재확인
    if args_new.val_id is not None:
        args.val_id = args_new.val_id
    if args_new.start_idx is not None:
        args.start_idx = args_new.start_idx
    if args_new.instruction is not None:
        args.instruction = args_new.instruction
    # start_idx 개수를 val_id 개수에 맞춤
    if len(args.start_idx) < len(args.val_id):
        args.start_idx = (args.start_idx * len(args.val_id))[: len(args.val_id)]
    if len(args.instruction) < len(args.val_id):
        args.instruction = (args.instruction * len(args.val_id))[: len(args.val_id)]
    # val_skip: CLI > config.down_sample fallback
    _val_skip = (
        args_new.val_skip
        if args_new.val_skip is not None
        else getattr(args, "down_sample", 1)
    )
    args.val_skip = _val_skip
    # interact_num: CLI 지정 시 config 덮어쓰기
    if args_new.interact_num is not None:
        args.interact_num = args_new.interact_num
    print(
        f"[CONFIG] task_type={args.task_type}  val_id={args.val_id}"
        f"  start_idx={args.start_idx}  val_skip={args.val_skip}"
        f"  interact_num={args.interact_num}  pred_step={args.pred_step}"
        f"  total_frames={args.interact_num * args.pred_step}  val_dataset_dir={args.val_dataset_dir}"
    )

    # CLI 플래그 → args 연결
    args.use_object_state = args_new.use_obj_token
    args.use_warning = args_new.use_warning
    args.adapter_ckpt = args_new.adapter_ckpt
    args.warning_guidance_scale = args_new.warning_guidance_scale
    args.obj_injection_scale = args_new.obj_scale
    args.warning_injection_scale = args_new.warn_scale
    args.annotation_only = args_new.annotation_only

    print(f"[SEED] {args.seed}")
    set_seed(args.seed)

    def upscale_frames(frames: list, scale: float) -> list:
        H_orig, W_orig = frames[0].shape[:2]
        H_new, W_new = int(H_orig * scale), int(W_orig * scale)
        result = []
        for f in frames:
            t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float()
            t = torch.nn.functional.interpolate(
                t,
                size=(H_new, W_new),
                mode="bilinear",
                align_corners=False,  # bilinear bicubic
            )
            result.append(t.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy())
        return result

    # ── Agent (pi0 + Ctrl-World) ────────────────────────────────
    Agent = agent(args)
    VIEW_IDX = args_new.view_idx

    # ── SAM3Enhancer ─────────────────────────────────────────────
    sam3_enhancer = SAM3Enhancer(
        mode=args_new.sam3_enhance_mode,
        scale=args_new.sam3_enhance_scale,
        realesrgan_root=args_new.realesrgan_root,
        realesrgan_model=args_new.realesrgan_model,
        device="cuda:0",
    )

    # ── SAM3 ────────────────────────────────────────────────────
    # sam_manager_orig: original frames only
    # sam_manager_enh:  enhanced/SR frames only (never mix domains in one manager)
    sam_manager_orig = SAM3Manager(checkpoint_path=args_new.sam3_ckpt, device="cuda:0")
    sam_manager_enh = None
    if sam3_enhancer.mode != "none":
        sam_manager_enh = SAM3Manager(
            checkpoint_path=args_new.sam3_ckpt, device="cuda:0"
        )
    sam_manager_up = None
    if args_new.upscale_scale is not None:
        sam_manager_up = SAM3Manager(
            checkpoint_path=args_new.sam3_ckpt, device="cuda:0"
        )

    # ── CLIP ────────────────────────────────────────────────────
    clip_model, clip_processor = None, None
    if args.clip_model_path and os.path.exists(str(args.clip_model_path)):
        from transformers import CLIPVisionModelWithProjection, CLIPProcessor

        print(f"Loading CLIP from {args.clip_model_path} ...")
        clip_model = CLIPVisionModelWithProjection.from_pretrained(
            args.clip_model_path
        ).to(Agent.device)
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
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    history_idx = args.history_idx

    colors = [
        (0, 255, 0),
        (0, 0, 255),
        (0, 150, 150),
        (0, 150, 255),
        (0, 255, 150),
        (0, 128, 200),
        (0, 128, 0),
    ]

    upscale_vis_to_save = []
    upscale_frames_to_save = []  # ← 이게 없음

    # ── trajectory 루프 ──────────────────────────────────────────
    for val_id_i, text_i, start_idx_i in zip(
        args.val_id, args.instruction, args.start_idx
    ):

        # ── GT trajectory + video_latents 로드 ───────────────────
        eef_gt, joint_pos_gt, video_dict, video_latents, _ = Agent.get_traj_info(
            val_id_i,
            start_idx=start_idx_i,
            steps=int(pred_step * interact_num + 8),
            skip=args.val_skip,
        )
        print(f"text_i: {text_i}  eef[0]: {eef_gt[0]}  joint[0]: {joint_pos_gt[0]}")

        # ── SAM3 초기화 ──────────────────────────────────────────
        first_frame = video_dict[VIEW_IDX][0]  # (H, W, 3) uint8

        if qwen_model is not None:
            object_labels = build_object_prompts_with_qwen(
                first_frame, text_i, qwen_model, qwen_processor, str(Agent.device)
            )
        elif args_new.object_labels:
            object_labels = [l.strip() for l in args_new.object_labels.split(",")]
        else:
            object_labels = ["robot arm and end-effector"]
        print(f"Object labels: {object_labels}")
        torch.cuda.empty_cache()

        # sam_manager.initialize 바로 위에 추가
        print("\n=== GPU Memory before SAM3 initialize ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(
                f"GPU {i}: allocated={allocated:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB"
            )
        print("==========================================\n")

        # ── 수동 마스크 → SAM3 point prompt 변환 ──────────────────
        _init_points_dict = None
        if args_new.init_masks_dir:
            _init_points_dict = {}
            for _li, _lbl in enumerate(object_labels):
                _mp = os.path.join(args_new.init_masks_dir, f"{_li}.png")
                if not os.path.exists(_mp):
                    print(f"[MASK] '{_lbl}': {_mp} 없음 → text prompt 사용")
                    continue
                _raw = cv2.imread(_mp, cv2.IMREAD_GRAYSCALE)
                if _raw is None:
                    print(f"[MASK] '{_lbl}': 읽기 실패 → text prompt 사용")
                    continue
                _m = _raw > 128
                _ys, _xs = np.where(_m)
                if len(_ys) == 0:
                    print(f"[MASK] '{_lbl}': 마스크 픽셀 없음 → text prompt 사용")
                    continue
                # 중심점 + 균등 샘플 최대 8점
                _cx, _cy = int(_xs.mean()), int(_ys.mean())
                _n = min(7, len(_xs))
                _idx = np.linspace(0, len(_xs) - 1, _n, dtype=int)
                _pts = [[_cx, _cy]] + [[int(_xs[i]), int(_ys[i])] for i in _idx]
                _init_points_dict[_lbl] = {
                    "point_coords": _pts,
                    "point_labels": [1] * len(_pts),
                }
                print(f"[MASK] '{_lbl}': {_mp} 로드 완료 → {len(_pts)}개 point prompt")

        _orig_H, _orig_W = first_frame.shape[:2]
        first_frame_sam = sam3_enhancer.enhance(first_frame)
        _init_points_dict_sam = sam3_enhancer.scale_points_up(_init_points_dict)

        # ── original-domain initialization (always) ──────────────────────
        sam_manager_orig.initialize(
            first_frame, object_labels, points_dict=_init_points_dict
        )
        init_domain_by_label = {}
        init_mask_by_label_orig = {}
        for label in object_labels:
            m = sam_manager_orig.object_masks.get(label)
            valid = m is not None and np.asarray(m).sum() > 0
            if valid:
                init_domain_by_label[label] = "original"
                init_mask_by_label_orig[label] = m
            else:
                init_domain_by_label[label] = None
                init_mask_by_label_orig[label] = None

        if sam3_enhancer.mode != "none":
            print(
                f"[SAM3Enhancer] first_frame enhanced: "
                f"{first_frame.shape[:2]} → {first_frame_sam.shape[:2]}"
            )

        # ── SR initialization (enhanced manager) ─────────────────────────
        # sam3_init_fallback_sr=True:  SR only for labels that failed in orig
        # sam3_init_fallback_sr=False: SR for all labels (enhanced-everything mode)
        # 단, manual mask가 있는 label은 플래그와 무관하게 항상 SR에서 detect:
        #   orig에서 point prompt로 잡히더라도 SR tracking이 더 안정적임
        _manual_mask_labels = (
            set(_init_points_dict.keys()) if _init_points_dict else set()
        )
        if sam_manager_enh is not None:
            if args_new.sam3_init_fallback_sr:
                _sr_init_labels = [
                    l
                    for l in object_labels
                    if init_domain_by_label.get(l) is None or l in _manual_mask_labels
                ]
            else:
                _sr_init_labels = list(object_labels)

            if _init_points_dict:
                _manual_labels = set(_init_points_dict.keys())
                _sr_init_labels = list(set(_sr_init_labels) | _manual_labels)
                print(
                    f"[SAM3 Init] Manual mask labels {list(_manual_labels)} "
                    f"added to SR detection (total: {_sr_init_labels})"
                )

            if _sr_init_labels:
                sam_manager_enh.initialize(
                    first_frame_sam, _sr_init_labels, points_dict=_init_points_dict_sam
                )
                for label in _sr_init_labels:
                    m_enh = sam_manager_enh.object_masks.get(label)
                    valid = m_enh is not None and np.asarray(m_enh).sum() > 0
                    if valid:
                        init_domain_by_label[label] = "super_resolution"
                        init_mask_by_label_orig[label] = resize_mask_to_original(
                            m_enh, _orig_H, _orig_W
                        )
                        print(f"[SAM3 enh init] '{label}': domain=super_resolution")
                    elif not args_new.sam3_init_fallback_sr:
                        print(
                            f"[SAM3 enh init] '{label}': enh failed, keep orig domain"
                        )

        preferred_detector_domain = dict(init_domain_by_label)
        print(f"[SAM3 init domain] {preferred_detector_domain}")

        # ── initial_areas in original pixel space ─────────────────────────
        _sam3_init_areas_orig = {
            lbl: float(m.sum()) if m is not None else 0.0
            for lbl, m in init_mask_by_label_orig.items()
        }

        # ---------------------------------- chuga
        initial_anchor_frame = first_frame.copy()
        initial_anchor_box_r = {}
        for label in object_labels:
            init_mask = init_mask_by_label_orig.get(label)
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
        # ------------------------------------
        # ── audit용 initial bbox area / bbox 저장 ───────────────────────
        initial_bbox_by_label = {}
        initial_bbox_area_by_label = {}

        for label in object_labels:
            init_mask = init_mask_by_label_orig.get(label)
            if init_mask is None:
                initial_bbox_by_label[label] = None
                initial_bbox_area_by_label[label] = 0.0
                continue

            _ys, _xs = np.where(init_mask)
            if len(_ys) == 0:
                initial_bbox_by_label[label] = None
                initial_bbox_area_by_label[label] = 0.0
                continue

            _x1, _y1, _x2, _y2 = (
                int(_xs.min()),
                int(_ys.min()),
                int(_xs.max()),
                int(_ys.max()),
            )
            initial_bbox_by_label[label] = [_x1, _y1, _x2, _y2]
            initial_bbox_area_by_label[label] = float(max(0, _x2 - _x1) * max(0, _y2 - _y1))
        # -------
        # 추가
        if sam_manager_up is not None:
            first_frame_up = upscale_frames([first_frame], args_new.upscale_scale)[0]
            sam_manager_up.initialize(first_frame_up, object_labels)

        # ── ObjectRegistry 초기화 ────────────────────────────────
        registry = ObjectRegistry()
        robot_label = object_labels[0]
        for label in object_labels:
            registry.register(label)

        initial_appearances = {}
        for label in object_labels:
            mask = init_mask_by_label_orig.get(label)
            _dom = init_domain_by_label.get(label) or "original"
            _sb = _dom != "original"
            _sf = float(sam3_enhancer.scale) if _sb else 1.0
            if mask is not None:
                app = (
                    registry.extract_appearance(
                        first_frame, mask, clip_model, clip_processor, str(Agent.device)
                    )
                    if clip_model is not None
                    else np.zeros(512, dtype=np.float32)
                )
                initial_appearances[label] = app
                bbox = ObjectRegistry.mask_to_bbox(mask, first_frame.shape[:2])
                shape_latent = ObjectRegistry.extract_shape_latent(mask)
                registry.update(
                    label,
                    presence=1.0,
                    appearance=app,
                    bbox=bbox,
                    state=0.0,
                    frame=first_frame,
                    mask=mask,
                    shape_latent=shape_latent,
                    detector_domain=_dom,
                    scale_back_applied=_sb,
                    scale_factor=_sf,
                )
            else:
                initial_appearances[label] = None
                registry.mark_absent(label)

        # ── history buffer 초기화 ────────────────────────────────
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(
            0
        )  # (1,4,72,40)
        assert first_latent.shape == (
            1,
            4,
            72,
            40,
        ), f"first_latent shape: {first_latent.shape}"

        his_cond = [first_latent] * (num_history * 4)
        his_joint = [joint_pos_gt[0:1]] * (num_history * 4)
        his_eef = [eef_gt[0:1]] * (num_history * 4)
        video_dict_pred = [v[0:1] for v in video_dict]  # 초기: GT 첫 프레임

        # ── 저장 버퍼 ────────────────────────────────────────────
        video_to_save = []
        vis_frames_to_save = []
        vel_vis_frames_to_save = []
        info_to_save = []
        tracking_log = {label: [] for label in object_labels}
        frame_counter = 0
        rollback_count = 0
        false_gen_count = 0

        # ── audit persistent memory across chunks ───────────────────────
        audit_last_good_by_label = {label: None for label in object_labels}
        audit_prev_bbox_by_label = {label: None for label in object_labels}

        # ── audit / Phase2 provenance metadata ─────────────────────────
        run_meta = {
            "traj_id": str(val_id_i),
            "task_id": str(val_id_i),
            "task_type": args.task_type,
            "task_name": getattr(args, "task_name", None),
            "text": text_i,
            "view_idx": int(VIEW_IDX),
            "seed": int(args.seed),
            "start_idx": int(start_idx_i),
            "val_skip": int(args.val_skip),
            "interact_num": int(args.interact_num),
            "pred_step": int(pred_step),
            "num_history": int(num_history),
            "sam3_enhance_mode": args_new.sam3_enhance_mode,
            "sam3_enhance_scale": int(args_new.sam3_enhance_scale),
            "object_labels": object_labels,
        }

        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        text_id = (
            text_i.replace(" ", "_")
            .replace(",", "")
            .replace(".", "")
            .replace("'", "")
            .replace('"', "")
        )[:40]
        base_dir = f"{args.save_dir}/{args.task_name}/video"
        os.makedirs(base_dir, exist_ok=True)

        # ── first frame + SAM3 초기 마스크 저장 (디버그용) ──────────
        _dbg_raw = os.path.join(
            base_dir, f"init_frame_traj{val_id_i}_view{VIEW_IDX}.png"
        )
        cv2.imwrite(_dbg_raw, first_frame[:, :, ::-1])  # RGB → BGR
        with open(
            os.path.join(base_dir, f"init_domain_traj{val_id_i}_view{VIEW_IDX}.json"),
            "w",
        ) as _jf:
            json.dump(init_domain_by_label, _jf, indent=2)
        _dbg_mask = first_frame.copy()
        for _ci, _lbl in enumerate(object_labels):
            _m = init_mask_by_label_orig.get(_lbl)
            if _m is not None:
                _c = colors[_ci % len(colors)]
                _dbg_mask[_m] = (_dbg_mask[_m] * 0.4 + np.array(_c) * 0.6).astype(
                    np.uint8
                )
                _ys, _xs = np.where(_m)
                if len(_ys):
                    _tx, _ty = int(_xs.mean()), int(_ys.mean())
                    cv2.putText(
                        _dbg_mask,
                        _lbl,
                        (_tx, _ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        _c,
                        1,
                        cv2.LINE_AA,
                    )
        _dbg_mask_path = os.path.join(
            base_dir, f"init_mask_traj{val_id_i}_view{VIEW_IDX}.png"
        )
        cv2.imwrite(_dbg_mask_path, _dbg_mask[:, :, ::-1])
        print(f"[DBG] first frame → {_dbg_raw}")
        print(f"[DBG] SAM3 init mask → {_dbg_mask_path}")
        # enhance debug: orig/enhanced frame + 각각의 mask overlay 저장
        if args_new.sam3_enhance_debug and sam3_enhancer.mode != "none":
            _enh_dbg_dir = os.path.join(
                args_new.sam3_enhance_cache_dir, f"traj{val_id_i}_init"
            )
            os.makedirs(_enh_dbg_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(_enh_dbg_dir, "first_frame_orig.png"),
                first_frame[:, :, ::-1],
            )
            cv2.imwrite(
                os.path.join(_enh_dbg_dir, "first_frame_enhanced.png"),
                first_frame_sam[:, :, ::-1],
            )

            # ── init_original_all_masks.png: sam_manager_orig 전체 결과 ──
            _dbg_orig_all = first_frame.copy()
            for _ci, _lbl in enumerate(object_labels):
                _m_orig = sam_manager_orig.object_masks.get(_lbl)
                if _m_orig is not None and np.asarray(_m_orig).sum() > 0:
                    _c = colors[_ci % len(colors)]
                    _dbg_orig_all[_m_orig] = (
                        _dbg_orig_all[_m_orig] * 0.4 + np.array(_c) * 0.6
                    ).astype(np.uint8)
                    _ys_o, _xs_o = np.where(_m_orig)
                    if len(_ys_o):
                        cv2.putText(
                            _dbg_orig_all,
                            _lbl[:20],
                            (int(_xs_o.mean()), int(_ys_o.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            _c,
                            1,
                            cv2.LINE_AA,
                        )
            cv2.imwrite(
                os.path.join(_enh_dbg_dir, "init_original_all_masks.png"),
                _dbg_orig_all[:, :, ::-1],
            )

            # ── init_enhanced_all_masks.png: sam_manager_enh 전체 결과 ──
            if sam_manager_enh is not None:
                _dbg_enh_all = first_frame_sam.copy()
                for _ci, _lbl in enumerate(object_labels):
                    _m_enh_all = sam_manager_enh.object_masks.get(_lbl)
                    if _m_enh_all is not None and np.asarray(_m_enh_all).sum() > 0:
                        _c = colors[_ci % len(colors)]
                        _dbg_enh_all[_m_enh_all] = (
                            _dbg_enh_all[_m_enh_all] * 0.4 + np.array(_c) * 0.6
                        ).astype(np.uint8)
                        _ys_e, _xs_e = np.where(_m_enh_all)
                        if len(_ys_e):
                            cv2.putText(
                                _dbg_enh_all,
                                _lbl[:20],
                                (int(_xs_e.mean()), int(_ys_e.mean())),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                _c,
                                1,
                                cv2.LINE_AA,
                            )
                cv2.imwrite(
                    os.path.join(_enh_dbg_dir, "init_enhanced_all_masks.png"),
                    _dbg_enh_all[:, :, ::-1],
                )

            # ── init_final_selected_masks_orig.png: registry에 들어가는 최종 mask ──
            _dbg_final = first_frame.copy()
            for _ci, _lbl in enumerate(object_labels):
                _m_final = init_mask_by_label_orig.get(_lbl)
                if _m_final is not None and np.asarray(_m_final).sum() > 0:
                    _c = colors[_ci % len(colors)]
                    _dbg_final[_m_final] = (
                        _dbg_final[_m_final] * 0.4 + np.array(_c) * 0.6
                    ).astype(np.uint8)
                    _ys_f, _xs_f = np.where(_m_final)
                    if len(_ys_f):
                        _dom_tag = init_domain_by_label.get(_lbl, "none")
                        cv2.putText(
                            _dbg_final,
                            f"{_lbl[:15]}[{_dom_tag[:3]}]",
                            (int(_xs_f.mean()), int(_ys_f.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            _c,
                            1,
                            cv2.LINE_AA,
                        )
            cv2.imwrite(
                os.path.join(_enh_dbg_dir, "init_final_selected_masks_orig.png"),
                _dbg_final[:, :, ::-1],
            )

            # ── enhanced_fallback_only_mask.png: SR fallback으로 선택된 label만 표시 ──
            _dbg_enh_fb = first_frame_sam.copy()
            for _ci, _lbl in enumerate(object_labels):
                _pref_d = init_domain_by_label.get(_lbl)
                if _pref_d == "super_resolution" and sam_manager_enh is not None:
                    _m_enh = sam_manager_enh.object_masks.get(_lbl)
                else:
                    _m_enh = None
                if _m_enh is not None:
                    _c = colors[_ci % len(colors)]
                    _dbg_enh_fb[_m_enh] = (
                        _dbg_enh_fb[_m_enh] * 0.4 + np.array(_c) * 0.6
                    ).astype(np.uint8)
            cv2.imwrite(
                os.path.join(_enh_dbg_dir, "enhanced_fallback_only_mask.png"),
                _dbg_enh_fb[:, :, ::-1],
            )

            print(f"[SAM3Enhancer] init debug → {_enh_dbg_dir}")
        if args_new.debug_first_frame:
            print("[DBG] --debug_first_frame: 저장 완료, 종료합니다.")
            import sys

            sys.exit(0)

        # ── SAM3 / rollback 상수 ─────────────────────────────────
        SAM3_CHUNK = 3
        REDETECT_SIM_THRESH = 0.70
        TRIM_MARGIN = args_new.rollback_trim_margin
        MAX_RETRIES = args_new.max_retries
        CAUSE_SEVERITY = {
            "occluded": 0.5,
            "out_of_frame": 0.7,
            "vanished": 1.0,
            "crushed": 1.5,
        }
        SOFT_RECOVER_MAX = 3
        DEFAULT_STREAK_THRESH = 2
        DEFAULT_SCORE_THRESH = 2.0
        ROBOT_STREAK_THRESH = 4
        ROBOT_SCORE_THRESH = 3.5

        bad_streak = {label: 0 for label in object_labels}
        error_score = {label: 0.0 for label in object_labels}
        soft_streak = {label: 0 for label in object_labels}

        # ── Phase2 training data accumulators (all frames, no trim) ──
        phase2_log = []  # list of per-frame dicts (WarningDataset format)
        phase2_latents = []  # list of (4, h, w) tensors for VIEW_IDX

        # ── Phase2DatasetBuilder 초기화 ──────────────────────────────
        _p2_builder = None
        if getattr(args_new, "phase2_data_dir", None):
            _p2_builder = Phase2DatasetBuilder(
                root=args_new.phase2_data_dir,
                config_snapshot={
                    "val_id": val_id_i,
                    "start_idx": start_idx_i,
                    "pred_step": pred_step,
                    "fps": args.fps,
                    "down_sample": getattr(args, "down_sample", 3),
                    "val_skip": args.val_skip,
                    "text": text_i,
                    "view_id": VIEW_IDX,
                },
            )

        # ── while 기반 rollout loop ──────────────────────────────
        i = 0
        step_retry_count = 0

        while i < interact_num:
            print(f"\n── Step {i+1}/{interact_num}  retry={step_retry_count} ──")

            # 스냅샷 (rollback 복원용)
            his_cond_snap = list(his_cond)
            his_eef_snap = list(his_eef)
            his_joint_snap = list(his_joint)
            frame_counter_snap = frame_counter
            registry_snap = registry.snapshot()
            video_dict_pred_snap = [v.copy() for v in video_dict_pred]
            snap_obj_state = registry.to_padded_tensor(N_OBJ).copy()
            snap_obj_shape = registry.to_padded_shape_tensor(N_OBJ).copy()

            step_full = []
            step_vis = []
            step_vel_vis = []
            step_log = {label: [] for label in object_labels}

            # retry 시 에러 점수 감쇠
            if step_retry_count > 0:
                for label in object_labels:
                    bad_streak[label] = int(bad_streak[label] * 0.5)
                    error_score[label] = round(error_score[label] * 0.5, 3)
            soft_streak = {label: 0 for label in object_labels}

            # GT video latents (forward_wm 비교용)
            start_id = int(i * (pred_step - 1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            # ── Policy forward (또는 annotation GT 사용) ─────────
            if args_new.annotation_only:
                # pi0 없이 GT annotation eef 직접 사용
                cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, 7)
                joint_pos = joint_pos_gt[start_id:end_id]  # (pred_step, 8)
                policy_in_out = {}
                print(
                    f"[annotation_only] cartesian_pose[0]={cartesian_pose[0]}  [-1]={cartesian_pose[-1]}"
                )
            else:
                print("### policy forward ###")
                current_joint = his_joint[-1][0]
                current_pose = his_eef[-1][0]
                current_obs = [v[-1] for v in video_dict_pred]
                policy_in_out, joint_pos, cartesian_pose = Agent.forward_policy(
                    current_obs, current_pose, current_joint, text=text_i
                )
                print(
                    f"cartesian_pose[0]={cartesian_pose[0]}  cartesian_pose[-1]={cartesian_pose[-1]}"
                )

            # ── World Model forward ───────────────────────────
            print("### world model forward ###")
            print(f"task: {text_i}, traj_id: {val_id_i}, step: {i}/{interact_num}")
            action_cond = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
            action_cond = np.concatenate([action_cond, cartesian_pose], axis=0)
            his_latent = torch.cat(
                [his_cond[idx] for idx in history_idx], dim=0
            ).unsqueeze(0)
            current_latent = his_cond[-1]

            obj_state_np = registry.to_padded_tensor(
                MAX_OBJECTS
            )  # (MAX_OBJECTS, FEATURE_DIM)
            obj_shape_np = registry.to_padded_shape_tensor(
                MAX_OBJECTS
            )  # (MAX_OBJECTS, SHAPE_SIZE^2)

            # warning_vec: inference 시점에는 현재 registry 상태로 soft 계산 (hard는 미래 미확인)
            warning_np = None
            if Agent.args.use_warning:
                from models.warning_utils import compute_warning_vec

                obj_records = []
                for lbl in registry.labels():
                    obj = registry.get(lbl)
                    obj_records.append(
                        {
                            "absent": obj.presence < 0.5,
                            "cause": None,  # inference 시 미래 cause 미확인
                            "bad_streak": 0,
                            "error_score": 0.0,
                            "iou": None,
                            "state": obj.state,
                            "shape_score": obj.shape_score,
                            "shape_rejected": obj.shape_rejected,
                            "area_ratio": obj.area_ratio,
                            "extent_ratio": obj.extent_ratio,
                            "bbox": obj.bbox.tolist(),
                        }
                    )
                warning_np = compute_warning_vec(obj_records)

            # ── phase1_eval_mode: 조건별 비교 후 바로 저장 ────────────
            if args_new.phase1_eval_mode:
                step_out_dir = os.path.join(
                    args_new.phase1_eval_out,
                    f"traj_{val_id_i}_start_{start_idx_i}",
                    f"step_{i}",
                )
                os.makedirs(step_out_dir, exist_ok=True)

                lat_ph1 = {}
                vid_ph1 = {}
                xtra_ph1 = {}

                for cname in args_new.phase1_eval_conditions:
                    cond_state, cond_shape = build_phase1_condition(
                        cname, registry, args_new, N_OBJ
                    )
                    _out = Agent.forward_wm(
                        action_cond,
                        video_latent_true,
                        current_latent,
                        his_cond=his_latent,
                        text=text_i if Agent.args.text_cond else None,
                        obj_state=cond_state,
                        obj_shape=cond_shape,
                        warning_vec=None,
                        condition_name=cname,
                        return_extra=True,
                    )
                    _vc, _tv, _vids, _lats, _extra = _out
                    lat_ph1[cname] = _lats  # (3, T, 4, h, w)
                    vid_ph1[cname] = _vids  # (3, T, H, W, 3)
                    xtra_ph1[cname] = _extra
                    mediapy.write_video(
                        os.path.join(step_out_dir, f"{cname}.mp4"),
                        _pack_views_horiz(_vids),
                        fps=args.fps,
                    )

                # diff 비디오
                _ref_v = vid_ph1.get("adapter_correct")
                if _ref_v is not None:
                    for _oth in ("adapter_zero", "adapter_shifted", "baseline_no_obj"):
                        if _oth in vid_ph1:
                            _save_diff_video(
                                _pack_views_horiz(_ref_v),
                                _pack_views_horiz(vid_ph1[_oth]),
                                os.path.join(
                                    step_out_dir, f"diff_correct_vs_{_oth}.mp4"
                                ),
                                fps=args.fps,
                            )
                if "adapter_zero" in vid_ph1 and "baseline_no_obj" in vid_ph1:
                    _save_diff_video(
                        _pack_views_horiz(vid_ph1["adapter_zero"]),
                        _pack_views_horiz(vid_ph1["baseline_no_obj"]),
                        os.path.join(step_out_dir, "diff_zero_vs_baseline_no_obj.mp4"),
                        fps=args.fps,
                    )

                # metrics.json
                _m = {
                    "adapter_ckpt": getattr(args_new, "adapter_ckpt", None),
                    "traj_id": val_id_i,
                    "start_idx": start_idx_i,
                    "step": i,
                    "text": text_i,
                    "object_labels": object_labels,
                    "conditions": args_new.phase1_eval_conditions,
                    # "val_dataset_dir": getattr(args_new, 'val_dataset_dir', None),
                    "annotation_path": f"{getattr(args_new, 'val_dataset_dir', '')}annotation/val/{val_id_i}.json",
                    "video_dir": f"{getattr(args_new, 'val_dataset_dir', '')}videos/val/{val_id_i}",
                }
                # per-condition extra
                for _cn, _ex in xtra_ph1.items():
                    for _k, _v in _ex.items():
                        if _v is not None:
                            _m[f"{_cn}_{_k}"] = _v

                # pairwise latent/pixel MSE
                _ref_l = lat_ph1.get("adapter_correct")
                _ref_vp = vid_ph1.get("adapter_correct")
                for _oth in ("adapter_zero", "adapter_shifted", "baseline_no_obj"):
                    if _ref_l is None or _oth not in lat_ph1:
                        continue
                    _m[f"latent_mse_correct_vs_{_oth}"] = _lat_mse(
                        _ref_l.float(), lat_ph1[_oth].float()
                    )
                    _m[f"pixel_mse_correct_vs_{_oth}"] = _px_mse(
                        _pack_views_horiz(_ref_vp), _pack_views_horiz(vid_ph1[_oth])
                    )
                if "adapter_zero" in lat_ph1 and "baseline_no_obj" in lat_ph1:
                    _m["latent_mse_zero_vs_baseline_no_obj"] = _lat_mse(
                        lat_ph1["adapter_zero"].float(),
                        lat_ph1["baseline_no_obj"].float(),
                    )
                    _m["pixel_mse_zero_vs_baseline_no_obj"] = _px_mse(
                        _pack_views_horiz(vid_ph1["adapter_zero"]),
                        _pack_views_horiz(vid_ph1["baseline_no_obj"]),
                    )

                # object region metrics — VIEW_IDX 기준만
                _reg_state = registry.to_padded_tensor(N_OBJ)
                _cur_bbox = _reg_state[:, 1:5]  # (MAX_OBJECTS, 4) bbox 인덱스
                _cur_pres = _reg_state[:, 0]  # (MAX_OBJECTS,)
                if _ref_vp is not None:
                    _vH, _vW = _ref_vp[VIEW_IDX].shape[1:3]
                    for _oth in ("adapter_zero", "adapter_shifted"):
                        if _oth not in vid_ph1:
                            continue
                        _rm = _region_metric_single_view(
                            _ref_vp[VIEW_IDX],
                            vid_ph1[_oth][VIEW_IDX],
                            _cur_bbox,
                            _cur_pres,
                            _vH,
                            _vW,
                        )
                        _m[f"object_region_mse_correct_vs_{_oth}"] = _rm["obj_mse"]
                        _m[f"background_region_mse_correct_vs_{_oth}"] = _rm["bg_mse"]
                        _m[f"obj_bg_ratio_correct_vs_{_oth}"] = _rm["ratio"]

                with open(os.path.join(step_out_dir, "metrics.json"), "w") as _f:
                    json.dump(_m, _f, indent=2)

                print(
                    f"[phase1_eval] traj={val_id_i} step={i}"
                    f"\n  c/z  lat={_m.get('latent_mse_correct_vs_adapter_zero',  'N/A')}"
                    f"  pix={_m.get('pixel_mse_correct_vs_adapter_zero',  'N/A')}"
                    f"\n  c/s  lat={_m.get('latent_mse_correct_vs_adapter_shifted','N/A')}"
                    f"  pix={_m.get('pixel_mse_correct_vs_adapter_shifted','N/A')}"
                    f"\n  c/b  lat={_m.get('latent_mse_correct_vs_baseline_no_obj','N/A')}"
                    f"\n  obj_bg_ratio c/z={_m.get('obj_bg_ratio_correct_vs_adapter_zero','N/A')}"
                    f"  c/s={_m.get('obj_bg_ratio_correct_vs_adapter_shifted','N/A')}"
                    f"\n  → {step_out_dir}"
                )

                if i + 1 >= args_new.phase1_eval_steps:
                    print(
                        f"[phase1_eval] phase1_eval_steps={args_new.phase1_eval_steps} 완료, 종료."
                    )
                    break
                i += 1
                continue
            # ── (phase1_eval_mode 아닐 때) 기존 forward_wm ─────────

            videos_cat, true_videos, video_dict_pred_new, predict_latents = (
                Agent.forward_wm(
                    action_cond,
                    video_latent_true,
                    current_latent,
                    his_cond=his_latent,
                    text=text_i if Agent.args.text_cond else None,
                    obj_state=obj_state_np,
                    obj_shape=obj_shape_np,
                    warning_vec=warning_np,
                )
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
                        upscale_frames(
                            [videos_cat[t] for t in range(n_save)],
                            args_new.upscale_scale,
                        )
                    )
                his_joint.append(joint_pos[pred_step - 1][None, :])
                his_eef.append(cartesian_pose[pred_step - 1][None, :])
                his_cond.append(
                    torch.cat(
                        [predict_latents[v][pred_step - 1] for v in range(3)], dim=1
                    ).unsqueeze(0)
                )
                video_dict_pred = video_dict_pred_new
                info_to_save.append(policy_in_out)
                i += 1
                continue

            # SAM3 tracking용 프레임
            pred_frames_track = [
                video_dict_pred_new[VIEW_IDX][t] for t in range(pred_step)
            ]
            pred_frames_full = [videos_cat[t] for t in range(pred_step)]

            # ── SAM3 Chunk Tracking ───────────────────────────
            # sam_manager_orig: original frames, sam_manager_enh: SR frames
            # SR frames are SAM3-input only — never passed to WM or used as latent
            pred_frames_sam = sam3_enhancer.enhance_list(pred_frames_track)

            # SR-domain metadata (for labels using sam_manager_enh)
            if sam3_enhancer.mode == "none":
                _detector_domain = "original"
                _scale_back_applied = False
                _detector_scale_factor = 1.0
            elif sam3_enhancer.mode == "realesrgan":
                _detector_domain = "super_resolution"
                _scale_back_applied = True
                _detector_scale_factor = float(sam3_enhancer.scale)
            else:
                _detector_domain = f"enhanced_{sam3_enhancer.mode}"
                _scale_back_applied = True
                _detector_scale_factor = float(sam3_enhancer.scale)

            _downstream_domain = "original"
            if sam3_enhancer.mode != "none":
                print(
                    f"[SAM3Enhancer] step {i}: {len(pred_frames_sam)} frames enhanced "
                    f"{pred_frames_track[0].shape[:2]} → {pred_frames_sam[0].shape[:2]}"
                )

            # labels by domain
            _enh_labels_step = [
                l
                for l in object_labels
                if preferred_detector_domain.get(l) == "super_resolution"
            ]

            sam_results_flat = []
            for chunk_start in range(0, len(pred_frames_track), SAM3_CHUNK):
                chunk_orig = pred_frames_track[chunk_start : chunk_start + SAM3_CHUNK]
                chunk_enh = pred_frames_sam[chunk_start : chunk_start + SAM3_CHUNK]

                # orig-domain: run on original frames for all labels
                chunk_results_orig = sam_manager_orig.update_chunk(chunk_orig)
                chunk_results_orig = [
                    sam3_frame_results_to_original(
                        fr,
                        _orig_H,
                        _orig_W,
                        detector_domain="original",
                        scale_back_applied=False,
                        scale_factor=1.0,
                        downstream_domain="original",
                    )
                    for fr in chunk_results_orig
                ]

                # enh-domain: run on SR frames for SR-domain labels only
                chunk_results_enh_raw = None
                if sam_manager_enh is not None and _enh_labels_step:
                    _cr_enh = sam_manager_enh.update_chunk(chunk_enh)
                    chunk_results_enh_raw = [
                        sam3_frame_results_to_original(
                            fr,
                            _orig_H,
                            _orig_W,
                            detector_domain=_detector_domain,
                            scale_back_applied=_scale_back_applied,
                            scale_factor=_detector_scale_factor,
                            downstream_domain="original",
                        )
                        for fr in _cr_enh
                    ]

                # merge: SR results override orig for SR-domain labels
                chunk_results = []
                for _tc in range(len(chunk_results_orig)):
                    fr_merged = dict(chunk_results_orig[_tc])
                    if chunk_results_enh_raw is not None:
                        for lbl in _enh_labels_step:
                            if lbl in chunk_results_enh_raw[_tc]:
                                fr_merged[lbl] = chunk_results_enh_raw[_tc][lbl]
                    chunk_results.append(fr_merged)

                sam_results_flat.extend(chunk_results)
                if any(
                    any(r.get("first_bad_t") is not None for r in fr.values())
                    for fr in chunk_results
                ):
                    break

            # enhance debug: original/enhanced/mask overlay 저장
            if (
                args_new.sam3_enhance_debug
                and sam3_enhancer.mode != "none"
                and sam_results_flat
            ):
                _enh_dbg_dir = os.path.join(
                    args_new.sam3_enhance_cache_dir, f"traj{val_id_i}_step{i}"
                )
                os.makedirs(_enh_dbg_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(_enh_dbg_dir, "orig_t0.png"),
                    pred_frames_track[0][:, :, ::-1],
                )
                cv2.imwrite(
                    os.path.join(_enh_dbg_dir, "enhanced_t0.png"),
                    pred_frames_sam[0][:, :, ::-1],
                )
                _vis_dbg = pred_frames_track[0].copy()
                for _ci, (_lbl, _res) in enumerate(sam_results_flat[0].items()):
                    _msk = _res.get("mask")
                    if _msk is not None:
                        _c = colors[_ci % len(colors)]
                        _vis_dbg[_msk] = (
                            _vis_dbg[_msk] * 0.5 + np.array(_c) * 0.5
                        ).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(_enh_dbg_dir, "mask_overlay_orig_t0.png"),
                    _vis_dbg[:, :, ::-1],
                )
                print(f"[SAM3Enhancer] step debug → {_enh_dbg_dir}")

            # upscale 병렬 tracking
            sam_results_flat_up = []
            if sam_manager_up is not None:
                pred_frames_track_up = upscale_frames(
                    pred_frames_track, args_new.upscale_scale
                )
                for chunk_start in range(0, len(pred_frames_track_up), SAM3_CHUNK):
                    chunk_up = pred_frames_track_up[
                        chunk_start : chunk_start + SAM3_CHUNK
                    ]
                    chunk_results_up = sam_manager_up.update_chunk(chunk_up)
                    sam_results_flat_up.extend(chunk_results_up)

            # ── 프레임별 처리 ─────────────────────────────────
            neg_event_detected = False
            first_bad_t = None
            _p2_bad_frame = None
            _p2_bad_latent = None
            _p2_bad_t = None  # intra-step frame index when failure was detected
            _p2_triggers = []
            _p2_causes = {}
            _p2_cur_obj_state = None
            _p2_cur_obj_shape = None

            for t, (frame_track, frame_full, sam_results) in enumerate(
                zip(pred_frames_track, pred_frames_full, sam_results_flat)
            ):
                redetected_labels = set()
                soft_recovered_labels = set()
                recovery_tier = {}
                rollback_candidate = {}

                for label in object_labels:
                    result = sam_results.get(label, {})
                    is_robot = label == robot_label

                    # robot arm: crushed 무시
                    if is_robot and result.get("cause") == "crushed":
                        sam_results[label]["cause"] = None
                        sam_results[label]["absent"] = False
                        result = sam_results[label]

                    if not (result.get("absent") or result.get("mask") is None):
                        bad_streak[label] = 0
                        soft_streak[label] = 0
                        error_score[label] = max(0.0, error_score[label] - 0.3)
                        recovery_tier[label] = 0
                        rollback_candidate[label] = False
                        continue

                    cause = result.get("cause") or "vanished"

                    # ── Tier 1: Soft recovery ─────────────────
                    if cause == "occluded" and soft_streak[label] < SOFT_RECOVER_MAX:
                        obj = registry.get(label)
                        # shape_score 낮으면 last_good_mask 신뢰도 우선
                        soft_mask = (
                            obj.last_good_mask.copy()
                            if obj.last_good_mask is not None
                            else None
                        )
                        if soft_mask is not None:
                            sam_results[label]["mask"] = soft_mask
                            sam_results[label]["absent"] = False
                            sam_results[label]["cause"] = None
                            soft_recovered_labels.add(label)
                        soft_streak[label] += 1
                        bad_streak[label] += 1
                        error_score[label] += CAUSE_SEVERITY.get("occluded", 0.5)
                        recovery_tier[label] = 1
                        st = ROBOT_STREAK_THRESH if is_robot else DEFAULT_STREAK_THRESH
                        sc = ROBOT_SCORE_THRESH if is_robot else DEFAULT_SCORE_THRESH
                        shape_bad = (not is_robot) and obj.shape_score < SHAPE_THRESH
                        rollback_candidate[label] = (
                            bad_streak[label] >= st
                            or error_score[label] >= sc
                            or shape_bad
                        )
                        print(
                            f"  [SOFT] '{label}': soft_streak={soft_streak[label]}/{SOFT_RECOVER_MAX}"
                            f"  streak={bad_streak[label]}  score={error_score[label]:.2f}"
                            f"  shape_score={obj.shape_score:.3f}  candidate={rollback_candidate[label]}"
                        )
                        continue

                    if cause == "occluded":
                        cause = "vanished"
                        sam_results[label]["cause"] = "vanished"
                        print(
                            f"  [ESCALATE] '{label}': soft_streak 초과 → vanished 승격"
                        )
                    soft_streak[label] = 0

                    # ── Tier 2: Re-detection ──────────────────
                    skip_redetect = cause == "crushed" and not is_robot
                    if not skip_redetect:
                        print(f"  [REDETECT] '{label}': cause={cause}, re-detecting...")
                        _pref_dom = preferred_detector_domain.get(label, "original")
                        if (
                            _pref_dom == "super_resolution"
                            and sam_manager_enh is not None
                        ):
                            _redet_manager = sam_manager_enh
                            _redet_frame = sam3_enhancer.enhance(frame_track)
                            _redet_sb = True
                        else:
                            _redet_manager = sam_manager_orig
                            _redet_frame = frame_track
                            _redet_sb = False
                        redet_mask_raw, _ = _redet_manager.redetect(_redet_frame, label)
                        redet_mask = (
                            resize_mask_to_original(redet_mask_raw, _orig_H, _orig_W)
                            if redet_mask_raw is not None and _redet_sb
                            else redet_mask_raw
                        )
                        if redet_mask is not None:
                            resumed = True
                            init_emb = initial_appearances.get(label)
                            if (
                                clip_model is not None
                                and init_emb is not None
                                and init_emb.sum() != 0
                            ):
                                curr_emb = registry.extract_appearance(
                                    frame_track,
                                    redet_mask,
                                    clip_model,
                                    clip_processor,
                                    str(Agent.device),
                                )
                                sim = float(
                                    np.dot(curr_emb, init_emb)
                                    / (
                                        np.linalg.norm(curr_emb)
                                        * np.linalg.norm(init_emb)
                                        + 1e-8
                                    )
                                )
                                print(
                                    f"  [REDETECT] '{label}': cosine_sim={sim:.4f}  thresh={REDETECT_SIM_THRESH}"
                                )
                                resumed = sim >= REDETECT_SIM_THRESH
                                if not resumed:
                                    sam_results[label]["cause"] = "vanished"
                                    cause = "vanished"
                            if resumed:
                                sam_results[label]["mask"] = redet_mask
                                sam_results[label]["absent"] = False
                                sam_results[label]["cause"] = None
                                redetected_labels.add(label)
                                bad_streak[label] = 0
                                error_score[label] = max(0.0, error_score[label] - 0.5)
                                recovery_tier[label] = 2
                                rollback_candidate[label] = False
                                print(
                                    f"  [REDETECT] '{label}': resumed  area={float(redet_mask.sum()):.0f}"
                                )
                                continue
                        else:
                            print(f"  [REDETECT] '{label}': not found → vanished")
                            sam_results[label]["cause"] = "vanished"
                            cause = "vanished"

                    # ── Tier 3: Hard failure ──────────────────
                    severity = CAUSE_SEVERITY.get(cause, 1.0)
                    bad_streak[label] += 1
                    error_score[label] += severity
                    sam_results[label]["absent"] = True
                    recovery_tier[label] = 3
                    st = ROBOT_STREAK_THRESH if is_robot else DEFAULT_STREAK_THRESH
                    sc = ROBOT_SCORE_THRESH if is_robot else DEFAULT_SCORE_THRESH
                    obj_hard = registry.get(label)
                    shape_bad = (not is_robot) and obj_hard.shape_score < SHAPE_THRESH
                    rollback_candidate[label] = (
                        bad_streak[label] >= st or error_score[label] >= sc or shape_bad
                    )
                    print(
                        f"  [SCORE] '{label}': cause={cause}"
                        f"  streak={bad_streak[label]}  score={error_score[label]:.2f}"
                        f"  shape_score={obj_hard.shape_score:.3f}  candidate={rollback_candidate[label]}"
                    )

                # rollback 판단 (non-robot 객체 기준)
                non_robot_trigger = [
                    lbl
                    for lbl in object_labels
                    if lbl != robot_label and rollback_candidate.get(lbl, False)
                ]
                if non_robot_trigger and not neg_event_detected:
                    neg_event_detected = True
                    first_bad_t = min(
                        max(0, t - bad_streak[lbl] + 1) for lbl in non_robot_trigger
                    )
                    print(
                        f"  [NEG] ROLLBACK triggered by {non_robot_trigger}  first_bad_t={first_bad_t}"
                    )
                    _p2_bad_frame = frame_track.copy()
                    _p2_bad_latent = predict_latents[VIEW_IDX][
                        t
                    ]  # (4, h, w) VIEW_IDX only
                    _p2_bad_t = t  # intra-step index
                    _p2_triggers = list(non_robot_trigger)
                    _p2_causes = {
                        lbl: sam_results.get(lbl, {}).get("cause")
                        for lbl in non_robot_trigger
                    }

                # ObjectRegistry 갱신
                interact_info = update_registry(
                    registry,
                    sam_results,
                    frame_track,
                    robot_label,
                    clip_model,
                    clip_processor,
                    str(Agent.device),
                    iou_thresh=args_new.iou_interaction,
                    initial_areas=_sam3_init_areas_orig,
                    detector_domain=_detector_domain,
                    scale_back_applied=_scale_back_applied,
                    scale_factor=_detector_scale_factor,
                )

                # Phase2: neg 이벤트가 이 프레임에서 처음 감지됐을 때 현재 obj state 캡처
                if _p2_bad_frame is not None and _p2_cur_obj_state is None:
                    _p2_cur_obj_state = registry.to_padded_tensor(N_OBJ).copy()
                    _p2_cur_obj_shape = registry.to_padded_shape_tensor(N_OBJ).copy()

                # tracking log
                for label in object_labels:
                    result = sam_results.get(label, {})
                    mask = result.get("mask")
                    area = float(mask.sum()) if mask is not None else 0.0
                    bbox = None
                    if mask is not None:
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            bbox = [
                                int(xs.min()),
                                int(ys.min()),
                                int(xs.max()),
                                int(ys.max()),
                            ]
                    H_f, W_f = frame_track.shape[:2]
                    bbox_norm = (
                        [bbox[0] / W_f, bbox[1] / H_f, bbox[2] / W_f, bbox[3] / H_f]
                        if bbox is not None
                        else [0.0, 0.0, 0.0, 0.0]
                    )
                    shape_lat = registry.get(label).to_shape_vector().tolist()
                    action_t = (
                        cartesian_pose[t].tolist()
                        if t < len(cartesian_pose)
                        else [0.0] * 7
                    )
                    appearance_sim = None
                    if clip_model is not None and mask is not None:
                        curr_emb = registry.extract_appearance(
                            frame_track,
                            mask,
                            clip_model,
                            clip_processor,
                            str(Agent.device),
                        )
                        init_emb = initial_appearances.get(label)
                        if init_emb is not None and init_emb.sum() != 0:
                            sim = float(
                                np.dot(curr_emb, init_emb)
                                / (
                                    np.linalg.norm(curr_emb) * np.linalg.norm(init_emb)
                                    + 1e-8
                                )
                            )
                            appearance_sim = round(sim, 4)
                            if result.get("cause") == "crushed":
                                print(
                                    f"  [APPEARANCE] '{label}' crushed: cosine_sim={sim:.4f}"
                                )
                    info = interact_info.get(label, {})
                    '''step_log[label].append({
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
                        "detector_domain":      _detector_domain,
                        "downstream_domain":    _downstream_domain,
                        "scale_back_applied":   _scale_back_applied,
                        "scale_factor":         _detector_scale_factor,
                    })
                    # ── audit last_good 계산: current append 전에 previous entries만 본다 ──
                    _prev_last_good = _find_last_good_entry(step_log[label])
                    _prev_entry = step_log[label][-1] if step_log[label] else None
                    _prev_bbox = (
                        _prev_entry.get("bbox") if _prev_entry is not None else None
                    )'''
                    _prev_last_good = audit_last_good_by_label.get(label)
                    _prev_bbox = audit_prev_bbox_by_label.get(label)

                    _bbox_stats = _bbox_stats_from_xyxy(
                        bbox,
                        init_bbox_area=initial_bbox_area_by_label.get(label, 0.0),
                        prev_bbox=_prev_bbox,
                    )

                    # 임시 entry를 먼저 만들어 update_reject_reason 계산에 사용
                    _entry_tmp = {
                        "absent": result.get("absent", False),
                        "rollback_candidate": rollback_candidate.get(label, False),
                        "shape_rejected": registry.get(label).shape_rejected,
                        "area_ratio": round(registry.get(label).area_ratio, 4),
                        "shape_score": round(registry.get(label).shape_score, 4),
                        "extent_ratio": round(registry.get(label).extent_ratio, 4),
                        "bbox": bbox,
                    }

                    _audit_good = _is_audit_good_frame(_entry_tmp)
                    _update_rejected = not _audit_good
                    _update_reject_reason = (
                        _audit_update_reject_reason(_entry_tmp)
                        if _update_rejected
                        else None
                    )

                    if _prev_last_good is not None:
                        _last_good_frame = _prev_last_good.get("frame")
                        _last_good_area = _prev_last_good.get("area")
                        _last_good_bbox = _prev_last_good.get("bbox")
                        _last_good_bbox_norm = _prev_last_good.get("bbox_norm")
                        _last_good_shape_latent = _prev_last_good.get("shape_latent")
                        _last_good_state = _prev_last_good.get("state")
                        _last_good_appearance = _prev_last_good.get("appearance")
                    else:
                        _last_good_frame = None
                        _last_good_area = None
                        _last_good_bbox = None
                        _last_good_bbox_norm = None
                        _last_good_shape_latent = None
                        _last_good_state = None
                        _last_good_appearance = None

                    _tracking_status = _tracking_status_from_entry(_entry_tmp)

                    _candidate_reasons = []
                    if result.get("absent", False):
                        _candidate_reasons.append("absent")
                    if rollback_candidate.get(label, False):
                        _candidate_reasons.append("rollback_candidate")
                    if registry.get(label).shape_rejected:
                        _candidate_reasons.append("shape_rejected")
                    if registry.get(label).shape_score < SHAPE_THRESH:
                        _candidate_reasons.append("shape_score_low")
                    if registry.get(label).area_ratio < AREA_THRESH:
                        _candidate_reasons.append("area_ratio_low")
                    if registry.get(label).extent_ratio < EXTENT_THRESH:
                        _candidate_reasons.append("extent_ratio_low")
                    _phase2_candidate = len(_candidate_reasons) > 0
                    _candidate_reason = "_".join(_candidate_reasons) if _candidate_reasons else None

                    _step_entry = {
                        "frame": frame_counter_snap + t,
                        "local_idx": (
                            int(start_id + t) if "start_id" in locals() else None
                        ),
                        # rollout/task provenance
                        "traj_id": str(val_id_i),
                        "task_id": str(val_id_i),
                        "task_type": args.task_type,
                        "text": text_i,
                        "view_idx": int(VIEW_IDX),
                        "seed": int(args.seed),
                        "start_idx": int(start_idx_i),
                        # object tracking state
                        "area": area,
                        "initial_area": float(_sam3_init_areas_orig.get(label, 0.0)),
                        "bbox": bbox,
                        "bbox_norm": bbox_norm,
                        "bbox_area": _bbox_stats["bbox_area"],
                        "bbox_area_ratio": _bbox_stats["bbox_area_ratio"],
                        "bbox_aspect": _bbox_stats["bbox_aspect"],
                        "bbox_center": _bbox_stats["bbox_center"],
                        "bbox_jump_px": _bbox_stats["bbox_jump_px"],
                        "shape_latent": shape_lat,
                        "action": action_t,
                        "absent": result.get("absent", False),
                        "cause": result.get("cause"),
                        "recovery_tier": recovery_tier.get(label, 0),
                        "rollback_candidate": rollback_candidate.get(label, False),
                        "redetected": label in redetected_labels,
                        "soft_recovered": label in soft_recovered_labels,
                        "bad_streak": bad_streak.get(label, 0),
                        "error_score": round(error_score.get(label, 0.0), 3),
                        "iou": info.get("iou"),
                        "state": info.get("state"),
                        "appearance": appearance_sim,
                        "shape_score": round(registry.get(label).shape_score, 4),
                        "shape_rejected": registry.get(label).shape_rejected,
                        "area_ratio": round(registry.get(label).area_ratio, 4),
                        "extent_ratio": round(registry.get(label).extent_ratio, 4),
                        # detector domain metadata
                        "detector_domain": _detector_domain,
                        "downstream_domain": _downstream_domain,
                        "scale_back_applied": _scale_back_applied,
                        "scale_factor": _detector_scale_factor,
                        # audit / last_good metadata
                        "last_good_frame": _last_good_frame,
                        "last_good_area": _last_good_area,
                        "last_good_bbox": _last_good_bbox,
                        "last_good_bbox_norm": _last_good_bbox_norm,
                        "last_good_shape_latent": _last_good_shape_latent,
                        "last_good_state": _last_good_state,
                        "last_good_appearance": _last_good_appearance,
                        "last_good_updated": bool(_audit_good),
                        "update_rejected": bool(_update_rejected),
                        "update_reject_reason": _update_reject_reason,
                        # phase2 candidate flag
                        "phase2_candidate": bool(_phase2_candidate),
                        "candidate_reason": _candidate_reason,
                        # human audit labels
                        "tracking_status": _tracking_status,
                        "generation_status": "unknown",
                        "audit_label": "todo",
                    }

                    step_log[label].append(_step_entry)

                    # ── persistent audit memory update ──────────────────────────────
                    audit_prev_bbox_by_label[label] = bbox

                    if _audit_good:
                        audit_last_good_by_label[label] = _step_entry

                # Phase2: ALL 프레임 누적 (trim 없이, do_rollback 여부 무관)
                if not args_new.raw_only:
                    frame_obj = {}
                    for lbl in object_labels:
                        if step_log[lbl]:
                            e = step_log[lbl][-1]
                            """frame_obj[lbl] = {
                                "absent": e["absent"],
                                "cause": e["cause"],
                                "bad_streak": e["bad_streak"],
                                "error_score": e["error_score"],
                                "iou": e["iou"],
                                "state": e["state"],
                                "shape_score": e["shape_score"],
                                "shape_rejected": e["shape_rejected"],
                                "area_ratio": e["area_ratio"],
                                "extent_ratio": e["extent_ratio"],
                                "bbox": e["bbox_norm"],
                                "appearance": [],
                                "shape_latent": e["shape_latent"],
                            }"""
                            frame_obj[lbl] = {
                                "absent": e["absent"],
                                "cause": e["cause"],
                                "bad_streak": e["bad_streak"],
                                "error_score": e["error_score"],
                                "iou": e["iou"],
                                "state": e["state"],
                                "shape_score": e["shape_score"],
                                "shape_rejected": e["shape_rejected"],
                                "area_ratio": e["area_ratio"],
                                "extent_ratio": e["extent_ratio"],

                                "bbox": e["bbox_norm"],
                                "bbox_px": e["bbox"],
                                "bbox_area": e["bbox_area"],
                                "bbox_area_ratio": e["bbox_area_ratio"],
                                "bbox_aspect": e["bbox_aspect"],
                                "bbox_center": e["bbox_center"],
                                "bbox_jump_px": e["bbox_jump_px"],

                                "appearance": [],
                                "shape_latent": e["shape_latent"],
                                # audit / last_good
                                "last_good_frame": e["last_good_frame"],
                                "last_good_area": e["last_good_area"],
                                "last_good_bbox": e["last_good_bbox"],
                                "last_good_updated": e["last_good_updated"],
                                "update_rejected": e["update_rejected"],
                                "update_reject_reason": e["update_reject_reason"],

                                # last_good extended fields
                                "last_good_bbox_norm": e["last_good_bbox_norm"],
                                "last_good_shape_latent": e["last_good_shape_latent"],
                                "last_good_state": e["last_good_state"],
                                "last_good_appearance": e["last_good_appearance"],
                                # phase2 candidate
                                "phase2_candidate": e["phase2_candidate"],
                                "candidate_reason": e["candidate_reason"],
                                # tracking audit
                                "tracking_status": e["tracking_status"],
                                "generation_status": e["generation_status"],
                                "audit_label": e["audit_label"],
                                # domain metadata
                                "detector_domain": e.get("detector_domain"),
                                "downstream_domain": e.get("downstream_domain"),
                                "scale_back_applied": e.get("scale_back_applied"),
                                "scale_factor": e.get("scale_factor"),
                            }

                    # local_idx: index into video_dict/video_latents for GT matching
                    # generated_frame[t] ↔ video_latents[VIEW_IDX][local_idx]
                    p2_local_idx = start_id + t
                    '''phase2_log.append(
                        {
                            "frame_idx": frame_counter_snap + t,
                            "local_idx": _p2_local_idx,
                            "action": action_t,
                            "objects": frame_obj,
                        }
                    )'''
                    phase2_log.append({
                        "frame_idx": frame_counter_snap + t,
                        "local_idx": p2_local_idx,
                        "traj_id":   str(val_id_i),
                        "task_id":   str(val_id_i),
                        "task_type": args.task_type,
                        "text":      text_i,
                        "view_idx":  int(VIEW_IDX),
                        "seed":      int(args.seed),
                        "start_idx": int(start_idx_i),
                        "action":    action_t,
                        "objects":   frame_obj,
                    })
                    phase2_latents.append(predict_latents[VIEW_IDX][t].cpu())

                # SAM3 overlay
                vis_frame = frame_track.copy()
                for ci, (label, result) in enumerate(sam_results.items()):
                    mask = result.get("mask")
                    if mask is None:
                        print(f"  [MASK FAIL] '{label}' tracking lost at t={t}")
                        continue
                    color = colors[ci % len(colors)]
                    vis_frame[mask] = (
                        vis_frame[mask] * 0.5 + np.array(color) * 0.5
                    ).astype(np.uint8)
                    ys, xs = np.where(mask)
                    if len(ys) > 0:
                        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
                        vis_frame[y1 : y1 + 2, x1:x2] = color
                        vis_frame[y2 : y2 + 2, x1:x2] = color
                        vis_frame[y1:y2, x1 : x1 + 2] = color
                        vis_frame[y1:y2, x2 : x2 + 2] = color
                        final_absent = (
                            result.get("absent")
                            or mask is None
                            or result.get("cause") in {"vanished", "crushed"}
                        )
                        if final_absent:
                            vis_frame[y1:y2, x1:x2] = (
                                vis_frame[y1:y2, x1:x2] * 0.5
                                + np.array([255, 0, 0]) * 0.5
                            ).astype(np.uint8)

                step_full.append(frame_full)
                step_vis.append(vis_frame)

                # ── velocity 시각화 프레임 ────────────────────────────
                vel_frame = vis_frame.copy()
                H_vf, W_vf = vel_frame.shape[:2]
                for ci, (label, result) in enumerate(sam_results.items()):
                    mask = result.get("mask")
                    color_bgr = colors[ci % len(colors)]  # (R,G,B) → cv2는 BGR
                    cv2_color = (
                        int(color_bgr[2]),
                        int(color_bgr[1]),
                        int(color_bgr[0]),
                    )

                    # centroid 점
                    _pref_dom_vis = preferred_detector_domain.get(label, "original")
                    if (
                        _pref_dom_vis == "super_resolution"
                        and sam_manager_enh is not None
                    ):
                        _vis_mgr = sam_manager_enh
                        _vis_s = sam3_enhancer.scale
                    else:
                        _vis_mgr = sam_manager_orig
                        _vis_s = 1
                    cent_hist = _vis_mgr._centroid_history.get(label, [])
                    valid_cents = [p for p in cent_hist if p is not None]
                    if valid_cents:
                        cx = int(valid_cents[-1][0] / _vis_s)
                        cy = int(valid_cents[-1][1] / _vis_s)
                        cv2.circle(vel_frame, (cx, cy), 5, (255, 255, 255), -1)
                        cv2.circle(vel_frame, (cx, cy), 5, cv2_color, 1)

                        # 속도 화살표 (최근 궤적 → 방향)
                        vel = _vis_mgr._centroid_velocity(label)
                        if vel is not None:
                            vx = vel[0] / _vis_s
                            vy = vel[1] / _vis_s
                            _arrow_s = 4.0
                            ex = int(cx + vx * _arrow_s)
                            ey = int(cy + vy * _arrow_s)
                            ex = max(0, min(W_vf - 1, ex))
                            ey = max(0, min(H_vf - 1, ey))
                            cv2.arrowedLine(
                                vel_frame,
                                (cx, cy),
                                (ex, ey),
                                (255, 255, 0),
                                2,
                                tipLength=0.4,
                            )

                        # 최근 centroid 궤적 (점선)
                        for pi in range(
                            max(0, len(valid_cents) - 5), len(valid_cents) - 1
                        ):
                            p1 = (
                                int(valid_cents[pi][0] / _vis_s),
                                int(valid_cents[pi][1] / _vis_s),
                            )
                            p2 = (
                                int(valid_cents[pi + 1][0] / _vis_s),
                                int(valid_cents[pi + 1][1] / _vis_s),
                            )
                            cv2.line(vel_frame, p1, p2, cv2_color, 1)

                    # border 접촉 엣지 강조 (노란색)
                    if mask is not None:
                        bm = 5
                        if mask[:bm, :].any():
                            cv2.line(vel_frame, (0, 0), (W_vf, 0), (0, 255, 255), 3)
                        if mask[-bm:, :].any():
                            cv2.line(
                                vel_frame,
                                (0, H_vf - 1),
                                (W_vf, H_vf - 1),
                                (0, 255, 255),
                                3,
                            )
                        if mask[:, :bm].any():
                            cv2.line(vel_frame, (0, 0), (0, H_vf), (0, 255, 255), 3)
                        if mask[:, -bm:].any():
                            cv2.line(
                                vel_frame,
                                (W_vf - 1, 0),
                                (W_vf - 1, H_vf),
                                (0, 255, 255),
                                3,
                            )

                    # cause 텍스트
                    cause = result.get("cause")
                    absent = result.get("absent", False)
                    if absent and cause:
                        cause_color = {
                            "occluded": (255, 165, 0),
                            "out_of_frame": (0, 255, 255),
                            "vanished": (0, 0, 255),
                            "crushed": (255, 0, 255),
                        }.get(cause, (255, 255, 255))
                        cause_cv2 = (
                            int(cause_color[2]),
                            int(cause_color[1]),
                            int(cause_color[0]),
                        )
                        if mask is not None:
                            ys_t, xs_t = np.where(mask)
                            tx = int(xs_t.min()) if len(ys_t) > 0 else 5
                            ty = int(ys_t.min()) - 6 if len(ys_t) > 0 else 15
                        else:
                            tx, ty = 5, 15 + ci * 15
                        ty = max(12, ty)
                        cv2.putText(
                            vel_frame,
                            cause,
                            (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            cause_cv2,
                            1,
                            cv2.LINE_AA,
                        )
                step_vel_vis.append(vel_frame)

                if sam_manager_up is not None and t < len(sam_results_flat_up):
                    sam_results_up = sam_results_flat_up[t]
                    frame_up = pred_frames_track_up[t]
                    vis_frame_up = frame_up.copy()
                    for ci, (label, result_up) in enumerate(sam_results_up.items()):
                        mask_up = result_up.get("mask")
                        if mask_up is None:
                            continue
                        color = colors[ci % len(colors)]
                        vis_frame_up[mask_up] = (
                            vis_frame_up[mask_up] * 0.5 + np.array(color) * 0.5
                        ).astype(np.uint8)
                        ys, xs = np.where(mask_up)
                        if len(ys) > 0:
                            y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
                            vis_frame_up[y1 : y1 + 2, x1:x2] = color
                            vis_frame_up[y2 : y2 + 2, x1:x2] = color
                            vis_frame_up[y1:y2, x1 : x1 + 2] = color
                            vis_frame_up[y1:y2, x2 : x2 + 2] = color
                            final_absent_up = result_up.get("absent") or result_up.get(
                                "cause"
                            ) in {"vanished", "crushed"}
                            if final_absent_up:
                                vis_frame_up[y1:y2, x1:x2] = (
                                    vis_frame_up[y1:y2, x1:x2] * 0.5
                                    + np.array([255, 0, 0]) * 0.5
                                ).astype(np.uint8)
                    upscale_vis_to_save.append(
                        vis_frame_up
                    )  # ← step 단위 임시 버퍼가 아닌 직접 누적

                if neg_event_detected:
                    break

            # ── Phase2 failure sample 저장 ────────────────────────
            if (
                _p2_bad_frame is not None
                and _p2_triggers
                and _p2_builder is not None
                and _p2_bad_t is not None
            ):
                _fg, _fc = classify_failure_group(_p2_triggers, _p2_causes, registry)
                _sid = f"{val_id_i}_{start_idx_i}_step{i}_r{step_retry_count}"

                # tracking_objects_info: loss region selector + negative descriptor only
                _tinfo = {}
                for _lbl in object_labels:
                    _obj_logs = step_log.get(_lbl, [])
                    if _obj_logs:
                        _e = _obj_logs[min(_p2_bad_t, len(_obj_logs) - 1)]
                        _tinfo[_lbl] = {
                            "bbox_norm": _e.get("bbox_norm", []),
                            "shape_score": _e.get("shape_score"),
                            "area_ratio": _e.get("area_ratio"),
                            "absent": _e.get("absent", False),
                            "shape_rejected": _e.get("shape_rejected", False),
                            "shape_latent": _e.get("shape_latent", []),
                            "preferred_detector_domain": preferred_detector_domain.get(
                                _lbl, "original"
                            ),
                        }

                # tracking_valid: object-level 조건 포함 종합 판정
                _tracking_valid = Phase2DatasetBuilder.compute_tracking_valid(
                    trigger_labels=_p2_triggers,
                    failure_group=_fg,
                    tracking_objects_info=_tinfo,
                )

                _p2_builder.save_failure_sample(
                    sample_id=_sid,
                    step_i=i,
                    start_id=start_id,  # = i * (pred_step - 1)
                    bad_t=_p2_bad_t,
                    view_id=VIEW_IDX,
                    bad_gen_frame=_p2_bad_frame,
                    bad_gen_latent=_p2_bad_latent,
                    video_dict=video_dict,
                    video_latents=video_latents,
                    registry=registry,
                    trigger_labels=_p2_triggers,
                    action_cond=action_cond,
                    history_latents=his_latent,
                    current_latent=current_latent,
                    failure_group=_fg,
                    failure_causes=_p2_causes,
                    object_labels=object_labels,
                    tracking_objects_info=_tinfo,
                    tracking_valid=_tracking_valid,
                    failure_detected=True,
                )

            # ── 롤백 vs 진행 결정 ────────────────────────────────
            do_rollback = (
                neg_event_detected
                and args_new.rollback_on_neg
                and step_retry_count < MAX_RETRIES
            )

            if do_rollback:
                trim_t = max(0, (first_bad_t or 0) - TRIM_MARGIN)
                print(
                    f"[ROLLBACK] step={i}  retry={step_retry_count+1}/{MAX_RETRIES}"
                    f"  first_bad_t={first_bad_t}  trim_t={trim_t}"
                )

                fg_frames = step_full[
                    (first_bad_t if first_bad_t is not None else trim_t) :
                ]
                if fg_frames:
                    fg_arr = np.stack(fg_frames, axis=0)
                    fg_path = (
                        f"{base_dir}/false_gen_{uuid}_{val_id_i}_{start_idx_i}"
                        f"_step{i}_r{step_retry_count}.mp4"
                    )
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
                his_cond = list(his_cond_snap)
                his_eef = list(his_eef_snap)
                his_joint = list(his_joint_snap)
                video_dict_pred = video_dict_pred_snap
                registry.restore(registry_snap)

                # rollback code 추가
                # --rollback_use_initial_real_frame
                for label in object_labels:
                    obj = registry.get(label)
                    # shape_score / area_ratio / extent_ratio 기준으로 anchor 선택
                    use_initial = (
                        obj.shape_score < SHAPE_THRESH
                        or obj.area_ratio < AREA_THRESH
                        or obj.extent_ratio < EXTENT_THRESH
                    )
                    if (
                        use_initial
                        and obj.initial_good_frame is not None
                        and obj.initial_good_mask is not None
                    ):
                        anchor_frame = obj.initial_good_frame
                        anchor_mask = obj.initial_good_mask
                        print(
                            f"  [ROLLBACK] '{label}': → initial anchor "
                            f"(shape={obj.shape_score:.2f} area={obj.area_ratio:.2f} ext={obj.extent_ratio:.2f})"
                        )
                    elif (
                        obj.last_good_frame is not None
                        and obj.last_good_mask is not None
                    ):
                        anchor_frame = obj.last_good_frame
                        anchor_mask = obj.last_good_mask
                        print(
                            f"  [ROLLBACK] '{label}': → last_good anchor "
                            f"(shape={obj.shape_score:.2f} area={obj.area_ratio:.2f} ext={obj.extent_ratio:.2f})"
                        )
                    else:
                        continue
                    ys, xs = np.where(anchor_mask)
                    H, W = anchor_frame.shape[:2]
                    if len(ys) > 0:
                        box_r = [
                            float(xs.min() / W),
                            float(ys.min() / H),
                            float(xs.max() / W),
                            float(ys.max() / H),
                        ]
                        _pref_dom = preferred_detector_domain.get(label, "original")
                        if (
                            _pref_dom == "super_resolution"
                            and sam_manager_enh is not None
                        ):
                            anchor_frame_sam = sam3_enhancer.enhance(anchor_frame)
                            sam_manager_enh.set_anchor(label, anchor_frame_sam, box_r)
                        else:
                            sam_manager_orig.set_anchor(label, anchor_frame, box_r)

                sam_manager_orig.reset_session()
                if sam_manager_enh is not None and _enh_labels_step:
                    sam_manager_enh.reset_session()

                """for label in object_labels:
                    obj = registry.get(label)
                    if obj.last_good_frame is not None and obj.last_good_mask is not None:
                        ys, xs = np.where(obj.last_good_mask)
                        H, W   = obj.last_good_frame.shape[:2]
                        if len(ys) > 0:
                            box_r = [float(xs.min()/W), float(ys.min()/H),
                                     float(xs.max()/W), float(ys.max()/H)]
                            sam_manager.set_anchor(label, obj.last_good_frame, box_r)
                sam_manager.reset_session()"""

                if sam_manager_up is not None:
                    sam_manager_up.reset_session()

                step_retry_count += 1
                rollback_count += 1
                continue

            # ── 진행 ─────────────────────────────────────────────
            step_retry_count = 0

            # rollback 없는 데이터 수집 모드: trim 없이 전 프레임 저장
            collect_mode = not args_new.rollback_on_neg
            if neg_event_detected and not collect_mode:
                trim_t = max(0, (first_bad_t or 0) - TRIM_MARGIN)
                n_save = trim_t
                print(
                    f"[MAX_RETRIES] step={i}: saving {n_save} safe frames and continuing"
                )
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
                torch.cat(
                    [predict_latents[v][pred_step - 1] for v in range(3)], dim=1
                ).unsqueeze(0)
            )
            video_dict_pred = video_dict_pred_new
            info_to_save.append(policy_in_out)

            for label in object_labels:
                bad_streak[label] = 0
                error_score[label] = 0.0
                soft_streak[label] = 0

            i += 1

        # ── 저장 ─────────────────────────────────────────────────
        print(
            "##########################################################################"
        )

        # 메인 비디오 (GT+pred side-by-side)
        filename_video = (
            f"{base_dir}/{args.task_type}_time_{uuid}_traj_{val_id_i}"
            f"_{start_idx_i}_{args.policy_skip_step}_{text_id}.mp4"
        )
        if args_new.upscale_scale is not None and upscale_frames_to_save:
            up_path = filename_video.replace(
                ".mp4", f"_up{args_new.upscale_scale}x.mp4"
            )
            mediapy.write_video(
                up_path, np.stack(upscale_frames_to_save, axis=0), fps=4 # fps가 길어지면, 몇 fps 마다 1초인지 딱 안 떨어질 때가 많아서, 일단 4로 고정함
            )
            print(f"Upscaled video: {up_path}")

        if video_to_save:
            mediapy.write_video(filename_video, np.stack(video_to_save, axis=0), fps=4) # fps가 길어지면, 몇 fps 마다 1초인지 딱 안 떨어질 때가 많아서, 일단 4로 고정함
            print(
                f"Saved: {filename_video}  (rollbacks={rollback_count}, false_gens={false_gen_count})"
            )

        # SAM3 overlay 영상
        if vis_frames_to_save:
            vis_path = filename_video.replace(".mp4", "_sam3.mp4")
            mediapy.write_video(vis_path, np.stack(vis_frames_to_save, axis=0), fps=4)
            print(f"SAM3 overlay: {vis_path}")

        # velocity 시각화 영상 (synthetic_traj/velocity/ 폴더)
        if vel_vis_frames_to_save:
            vel_dir = os.path.join(args.save_dir, "velocity")
            os.makedirs(vel_dir, exist_ok=True)
            vel_filename = os.path.basename(filename_video).replace(".mp4", "_vel.mp4")
            vel_path = os.path.join(vel_dir, vel_filename)
            mediapy.write_video(
                vel_path, np.stack(vel_vis_frames_to_save, axis=0), fps=4
            )
            print(f"Velocity vis: {vel_path}")

        if upscale_vis_to_save:
            up_vis_path = filename_video.replace(
                ".mp4", f"_sam3_up{args_new.upscale_scale}x.mp4"
            )
            mediapy.write_video(
                up_vis_path, np.stack(upscale_vis_to_save, axis=0), fps=4
            )
            print(f"SAM3 upscale overlay: {up_vis_path}")

        # 객체별 tracking log JSON
        for label, log in tracking_log.items():
            safe_label = label.replace(" ", "_").replace("/", "-")
            json_path = filename_video.replace(".mp4", f"_{safe_label}.json")
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "label": label,
                        "initial_area": _sam3_init_areas_orig.get(label, 0.0),
                        "frames": log,
                    },
                    f,
                    indent=2,
                )
            print(f"Tracking log: {json_path}")

        # Phase2 training data (tracking.json + generated latent + GT latent)
        if phase2_log and phase2_latents:
            phase2_dir = filename_video.replace(".mp4", "_phase2")
            os.makedirs(phase2_dir, exist_ok=True)

            # config snapshot: frame index 복원에 필요한 모든 정보 포함
            '''phase2_meta = {
                "episode_id": os.path.basename(phase2_dir),
                "object_labels": object_labels,
                "language_instruction": text_i,
                # config snapshot
                "val_id": val_id_i,
                "start_idx": start_idx_i,
                "pred_step": pred_step,
                "fps": args.fps,
                "down_sample": getattr(args, "down_sample", 3),
                "val_skip": args.val_skip,
                "frame_stride": args.val_skip,
                "view_id": VIEW_IDX,
                # frames (tracking info: loss region selector + negative descriptor ONLY)
                "frames": phase2_log,
                "NOTE_frames": "bbox/shape_latent/area_ratio for loss_region_selector "
                "and negative_descriptor only — NOT positive training target",
            }'''
            phase2_meta = {
                "episode_id":           os.path.basename(phase2_dir),
                "object_labels":        object_labels,
                "language_instruction": text_i,

                # rollout/task metadata
                "traj_id":              str(val_id_i),
                "task_id":              str(val_id_i),
                "task_type":            args.task_type,
                "task_name":            getattr(args, "task_name", None),
                "text":                 text_i,
                "view_idx":             int(VIEW_IDX),
                "seed":                 int(args.seed),
                "start_idx":            int(start_idx_i),
                "val_skip":             int(args.val_skip),
                "interact_num":         int(args.interact_num),
                "pred_step":            int(pred_step),
                "num_history":          int(num_history),

                # file provenance
                "video_path":           filename_video,
                "phase2_dir":           phase2_dir,

                # SAM3 / enhancement metadata
                "sam3_enhance_mode":    args_new.sam3_enhance_mode,
                "sam3_enhance_scale":   int(args_new.sam3_enhance_scale),
                "init_domain_by_label": init_domain_by_label,
                "initial_area_by_label": _sam3_init_areas_orig,
                "initial_bbox_by_label": initial_bbox_by_label,
                "initial_bbox_area_by_label": initial_bbox_area_by_label,

                # human audit policy
                "audit_label_set": [
                    "tracking_ok_generation_ok",
                    "tracking_ok_generation_bad",
                    "tracking_bad_generation_unknown",
                    "tracking_lost",
                    "uncertain",
                ],
                "phase2_use_policy": "Use only frames/episodes manually verified as tracking_ok_generation_bad.",
                "phase2_candidate_policy": {
                    "candidate_is_not_training_label": True,
                    "use_for_training_only_if_audit_label": "tracking_ok_generation_bad",
                    "exclude_labels": [
                        "tracking_bad_generation_unknown",
                        "tracking_lost",
                        "uncertain",
                    ],
                },

                "frames":               phase2_log,
        }
            with open(os.path.join(phase2_dir, "tracking.json"), "w") as f:
                json.dump(phase2_meta, f, indent=2)

            # generated latents (bad/unknown quality — input or negative)
            latent_tensor = torch.stack(phase2_latents, dim=0)  # (T, 4, h, w)
            torch.save(latent_tensor, os.path.join(phase2_dir, "latent.pt"))

            # GT real latents: match each generated frame to the corresponding GT
            # generated frame[t] ↔ video_latents[VIEW_IDX][local_idx]
            gt_latents = []
            gt_frames = []
            n_gt = video_dict[VIEW_IDX].shape[0]
            for _entry in phase2_log:
                _li = _entry.get("local_idx", _entry["frame_idx"])  # prefer local_idx
                if _li < n_gt:
                    gt_latents.append(video_latents[VIEW_IDX][_li].cpu())
                    gt_frames.append(video_dict[VIEW_IDX][_li])
                else:
                    gt_latents.append(
                        torch.zeros_like(video_latents[VIEW_IDX][0].cpu())
                    )
                    gt_frames.append(np.zeros_like(video_dict[VIEW_IDX][0]))
            gt_latent_tensor = torch.stack(gt_latents, dim=0)  # (T, 4, h, w)
            torch.save(gt_latent_tensor, os.path.join(phase2_dir, "gt_latent.pt"))
            np.save(
                os.path.join(phase2_dir, "gt_frames.npy"), np.stack(gt_frames, axis=0)
            )

            print(
                f"Phase2 data: {phase2_dir}  ({len(phase2_log)} frames, "
                f"gen_latent={tuple(latent_tensor.shape)}  "
                f"gt_latent={tuple(gt_latent_tensor.shape)})"
            )

        # policy info JSON
        if info_to_save:
            info = {
                "success": 1,
                "start_idx": 0,
                "end_idx": len(video_to_save) - 1,
                "instructions": text_i,
            }
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
            with open(filename_info, "w") as f:
                json.dump(info, f, indent=4)
            print(f"Policy info: {filename_info}")

        print(
            "##########################################################################"
        )


# 실행 예:

"""CUDA_VISIBLE_DEVICES=0,1 python scripts/rollout_interact_pi_online.py \
  --task_type pickplace \
  --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt \
  --svd_model_path checkpoints/svd \
  --clip_model_path checkpoints/clip \
  --pi_ckpt /home/dgu/minyoung/checkpoints/pi05_droid_pytorch \
  --sam3_ckpt /home/dgu/minyoung/sam3/checkpoints/sam3.pt \
  --object_labels "robot arm and end-effector,pen" \
  --view_idx 1 \
  --seed 42 """

'''python scripts/rollout_interact_pi_online.py \
  --task_type pickplace --seed 42 \
  --ckpt_path /path/to/finetuned.pt \
  --use_obj_token \
  --object_labels "pen"'''

"""# SAM3 enhance mode (realesrgan)
CUDA_VISIBLE_DEVICES=0,1 python scripts/rollout_interact_pi_online.py \
  --task_type pickplace \
  --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt \
  --svd_model_path checkpoints/svd \
  --clip_model_path checkpoints/clip \
  --pi_ckpt /home/dgu/minyoung/checkpoints/pi05_droid_pytorch \
  --sam3_ckpt /home/dgu/minyoung/sam3/checkpoints/sam3.pt \
  --object_labels "robot arm and end-effector,pen" \
  --view_idx 1 \
  --seed 10 \
  --sam3_enhance_mode realesrgan \
  --sam3_enhance_scale 4 \
  --realesrgan_root /home/dgu/minyoung/Real-ESRGAN \
  --sam3_enhance_debug"""
