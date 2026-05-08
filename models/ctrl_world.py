# from diffusers import StableVideoDiffusionPipeline
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from models.object_registry import FEATURE_DIM, MAX_OBJECTS, SHAPE_SIZE
from models.warning_utils import WARNING_DIM

OBJ_INJECTION_SCALE = 0.1
WARNING_INJECTION_SCALE = 0.1
 #
import numpy as np
import torch
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import wandb
#import swanlab
import mediapy


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Action_encoder2(nn.Module): # trajetory encoder, encode action sequence to a hidden vector, support text condition
    def __init__(self, action_dim, action_num, hidden_size, text_cond=True):
        super().__init__()
        self.action_dim = action_dim # 액션 차원 (예: 7 = xyz+quat or 6D euler+gripper)
        self.action_num = action_num # 총 프레임 수 (history + future)
        self.hidden_size = hidden_size # 출력 차원 (1024)
        self.text_cond = text_cond # 텍스트 조건 사용 여부

        input_dim = int(action_dim)
        self.action_encode = nn.Sequential(
            nn.Linear(input_dim, 1024), # action_dim → 1024
            nn.SiLU(), # Swish 활성화 함수 (부드러운 ReLU)
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024) # 최종 출력: 1024차원
        )
        # kaiming initialization
        nn.init.kaiming_normal_(self.action_encode[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.action_encode[2].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, action,  texts=None, text_tokinizer=None, text_encoder=None, frame_level_cond=True,):
        # action: (B, action_num, action_dim)
        B,T,D = action.shape
        if not frame_level_cond:
            action = einops.rearrange(action, 'b t d -> b 1 (t d)')
        action = self.action_encode(action) # MLP 통과 → (B, T, 1024) or (B, 1, 1024)

        if texts is not None and self.text_cond:
            # with 50% probability, add text condition
            with torch.no_grad():
                inputs = text_tokinizer(texts, padding='max_length', return_tensors="pt", truncation=True).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                hidden_text = outputs.text_embeds # (B, 512) # CLIP text embed: (B, 512)
                hidden_text = einops.repeat(hidden_text, 'b c -> b 1 (n c)', n=2) # (B, 1, 1024) # (B, 1, 1024) — action과 차원 맞춤

            action = action + hidden_text # (B, T, hidden_size)
        return action # (B, 1, hidden_size) or (B, T, hidden_size) if frame_level_cond


SHAPE_PROJ_DIM = 64   # shape_latent (256) → projected dim before concat with obj_state


class WarningEncoder(nn.Module):
    """
    warning_vec (B, 8) → (B, 1, 1024) warning token.
    Linear(8,64) → SiLU → Linear(64,128) → Linear(128,1024).
    Zero-init on final projection for backward compatibility.
    """
    def __init__(self, warning_dim: int = WARNING_DIM, hidden_size: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(warning_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
        )
        self.proj = nn.Linear(128, hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, warning_vec: torch.Tensor) -> torch.Tensor:
        """warning_vec: (B, 8) → (B, 1, 1024)"""
        return self.proj(self.encoder(warning_vec)).unsqueeze(1)


class ShapeProjector(nn.Module):
    """
    shape_latent (B, MAX_OBJECTS, SHAPE_SIZE^2) → (B, MAX_OBJECTS, SHAPE_PROJ_DIM).
    경량 MLP — ObjectStateEncoder 입력 전 shape 정보를 압축.
    """
    def __init__(self, shape_dim: int = SHAPE_SIZE * SHAPE_SIZE, proj_dim: int = SHAPE_PROJ_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(shape_dim, 128),
            nn.SiLU(),
            nn.Linear(128, proj_dim),
        )

    def forward(self, obj_shape: torch.Tensor) -> torch.Tensor:
        """obj_shape: (B, MAX_OBJECTS, SHAPE_SIZE^2) → (B, MAX_OBJECTS, SHAPE_PROJ_DIM)"""
        return self.proj(obj_shape)


class ObjectStateEncoder(nn.Module):
    """
    [obj_state ‖ obj_shape_proj] → (B, MAX_OBJECTS, 1024) token sequence.

    입력:  (B, MAX_OBJECTS, FEATURE_DIM + SHAPE_PROJ_DIM)
    출력:  (B, MAX_OBJECTS, hidden_size)

    학습 전략 (classifier-free guidance 방식):
      - training 중 obj_dropout_prob 확률로 전체 token을 zeros로 치환
        → 모델이 object token 없이도 동작하도록 학습 (기존 ckpt 호환)
      - inference 시 실제 object state를 주입하면 conditioning 효과 발생
    """
    def __init__(self, feature_dim: int = FEATURE_DIM + SHAPE_PROJ_DIM,
                 hidden_size: int = 1024, obj_dropout_prob: float = 0.5):
        super().__init__()
        self.obj_dropout_prob = obj_dropout_prob
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, hidden_size),
        )
        # zero init → 학습 초반에 기존 action token에 영향 최소화
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, obj_feat: torch.Tensor) -> torch.Tensor:
        """
        obj_feat: (B, MAX_OBJECTS, FEATURE_DIM + SHAPE_PROJ_DIM) float
        returns:  (B, MAX_OBJECTS, 1024)
        """
        result = self.encoder(obj_feat)  # 항상 계산 — grad_fn 유지
        if self.training and torch.rand(1).item() < self.obj_dropout_prob:
            return result * 0.0  # zeros이지만 grad_fn 살아있음 (CFG dropout)
        return result


class CrtlWorld(nn.Module): # main model class, input action sequence and text, output future frame latent prediction, support frame_level pose condition
    def __init__(self, args):
        super(CrtlWorld, self).__init__()

        self.args = args

        # load from pretrained stable video diffusion
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(args.svd_model_path)
        # repalce the unet to support frame_level pose condition
        print("replace the unet to support action condition and frame_level pose!")
        unet = UNetSpatioTemporalConditionModel()
        unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)
        self.pipeline.unet = unet

        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.image_encoder = self.pipeline.image_encoder
        self.scheduler = self.pipeline.scheduler

        # freeze all backbone — only shape_projector / object_state_encoder are trainable
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()

        # SVD is a img2video model, load a clip text encoder
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.clip_model_path,use_fast=False)
        self.text_encoder.requires_grad_(False)

        # initialize an action projector
        self.action_encoder = Action_encoder2(action_dim=args.action_dim, action_num=int(args.num_history+args.num_frames), hidden_size=1024, text_cond=args.text_cond)
        self.action_encoder.requires_grad_(False)

        self.use_object_state = getattr(args, 'use_object_state', False)

        if self.use_object_state:
            self.shape_projector = ShapeProjector(
                shape_dim=SHAPE_SIZE * SHAPE_SIZE,
                proj_dim=SHAPE_PROJ_DIM,
            )
            self.object_state_encoder = ObjectStateEncoder(
                feature_dim=FEATURE_DIM + SHAPE_PROJ_DIM,
                hidden_size=1024,
                obj_dropout_prob=getattr(args, 'obj_dropout_prob', 0.5),
            )
        else:
            self.shape_projector       = None
            self.object_state_encoder  = None

        self.use_warning = getattr(args, 'use_warning', False)
        if self.use_warning:
            self.warning_encoder = WarningEncoder(warning_dim=WARNING_DIM, hidden_size=1024)
        else:
            self.warning_encoder = None



    def forward(self, batch):
        latents = batch['latent'] # (B, 16, 4, 32, 32)
        texts = batch['text']
        dtype = self.unet.dtype
        device = self.unet.device
        P_mean=0.7
        P_std=1.6
        noise_aug_strength = 0.0

        num_history  = self.args.num_history
        latents = latents.to(device) #[B, num_history + num_frames]

        # current img as condition image to stack at channel wise, add random noise to current image, noise strength 0.0~0.2
        current_img = latents[:,num_history:(num_history+1)] # (B, 1, 4, 32, 32)
        bsz,num_frames = latents.shape[:2]
        current_img = current_img[:,0] # (B, 4, 32, 32)
        sigma = torch.rand([bsz, 1, 1, 1], device=device) * 0.2
        c_in = 1 / (sigma**2 + 1) ** 0.5
        current_img = c_in*(current_img + torch.randn_like(current_img) * sigma)
        condition_latent = einops.repeat(current_img, 'b c h w -> b f c h w', f=num_frames) # (8, 16,12, 32,32)
        if self.args.his_cond_zero:
            condition_latent[:, :num_history] = 0.0 # (B, num_history+num_frames, 4, 32, 32)


        # action condition
        action = batch['action'] # (B, f, 7)
        action = action.to(device)
        action_hidden = self.action_encoder(action, texts, self.tokenizer, self.text_encoder, frame_level_cond=self.args.frame_level_cond) # (B, f, 1024)

        # for classifier-free guidance, with 5% probability, set action_hidden to 0
        uncond_hidden_states = torch.zeros_like(action_hidden)
        text_mask = (torch.rand(action_hidden.shape[0], device=device)>0.05).unsqueeze(1).unsqueeze(2)
        action_hidden = action_hidden*text_mask+uncond_hidden_states*(~text_mask)

        # object-state + shape token conditioning
        if self.object_state_encoder is not None:
            obj_state = (batch['obj_state'].to(device).to(dtype)
                         if 'obj_state' in batch and batch['obj_state'] is not None
                         else torch.zeros(bsz, MAX_OBJECTS, FEATURE_DIM, device=device, dtype=dtype))
            obj_shape = (batch['obj_shape'].to(device).to(dtype)
                         if 'obj_shape' in batch and batch['obj_shape'] is not None
                         else torch.zeros(bsz, MAX_OBJECTS, SHAPE_SIZE * SHAPE_SIZE, device=device, dtype=dtype))
            obj_shape_proj = self.shape_projector(obj_shape)           # (B, MAX_OBJECTS, SHAPE_PROJ_DIM)
            obj_feat       = torch.cat([obj_state, obj_shape_proj], dim=-1)  # (B, MAX_OBJECTS, FEATURE_DIM+SHAPE_PROJ_DIM)
            obj_tokens    = self.object_state_encoder(obj_feat)        # (B, MAX_OBJECTS, 1024)
            # residual injection: 시퀀스 길이 유지
            n_obj = min(obj_tokens.shape[1], action_hidden.shape[1])
            action_hidden = action_hidden.clone()
            _obj_scale_inj = getattr(self.args, 'obj_injection_scale', OBJ_INJECTION_SCALE)
            action_hidden[:, :n_obj] = action_hidden[:, :n_obj] + _obj_scale_inj * obj_tokens[:, :n_obj]

        # diffusion forward process on future latent
        rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        c_skip = 1 / (sigma**2 + 1)
        c_out =  -sigma / (sigma**2 + 1) ** 0.5
        c_in = 1 / (sigma**2 + 1) ** 0.5
        c_noise = (sigma.log() / 4).reshape([bsz])
        loss_weight = (sigma ** 2 + 1) / sigma ** 2
        noisy_latents = (latents + torch.randn_like(latents) * sigma)

        # add 0~0.3 noise to history, history as condition
        sigma_h = torch.randn([bsz, num_history, 1, 1, 1], device=device) * 0.3
        history = latents[:,:num_history] # (B, num_history, 4, 32, 32)
        noisy_history = 1/(sigma_h**2+1)**0.5 *(history + sigma_h * torch.randn_like(history)) # (B, num_history, 4, 32, 32)
        input_latents = torch.cat([noisy_history, c_in*noisy_latents[:,num_history:]], dim=1) # (B, num_history+num_frames, 4, 32, 32)

        # svd stack a img at channel wise
        input_latents = torch.cat([input_latents, condition_latent/self.vae.config.scaling_factor], dim=2)
        motion_bucket_id = self.args.motion_bucket_id
        fps = self.args.fps
        added_time_ids = self.pipeline._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, action_hidden.dtype, bsz, 1, False)
        added_time_ids = added_time_ids.to(device)

        # warning token conditioning
        if self.warning_encoder is not None:
            warning_vec = (batch['warning_vec'].to(device).to(dtype)
                           if 'warning_vec' in batch and batch['warning_vec'] is not None
                           else torch.zeros(bsz, WARNING_DIM, device=device, dtype=dtype))
            warning_token = self.warning_encoder(warning_vec)     # (B, 1, 1024)
            action_hidden = action_hidden.clone()
            _warn_scale_inj = getattr(self.args, 'warning_injection_scale', WARNING_INJECTION_SCALE)
            action_hidden[:, :1] = action_hidden[:, :1] + _warn_scale_inj * warning_token  # 첫 토큰에 residual

        # forward unet
        loss = 0
        model_pred = self.unet(input_latents, c_noise, encoder_hidden_states=action_hidden, added_time_ids=added_time_ids,frame_level_cond=self.args.frame_level_cond).sample
        predict_x0 = c_out * model_pred + c_skip * noisy_latents

        # only calculate loss on future frames
        loss += ((predict_x0[:,num_history:] - latents[:,num_history:])**2 * loss_weight).mean()

        return loss, torch.tensor(0.0, device=device,dtype=dtype)
