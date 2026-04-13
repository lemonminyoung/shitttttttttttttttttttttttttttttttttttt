"""
rollout_online.py — Object-Aware Online Generation Pipeline

파이프라인 흐름:
  [Initialization]
  1. get_frames()로 첫 프레임 로드
  2. Qwen3-VL → 장면 내 객체 목록 + SAM3 텍스트 프롬프트 생성
  3. SAM3Manager.initialize() → 첫 프레임에서 객체별 mask 탐지
  4. ObjectRegistry 초기화 (appearance=CLIP embedding, bbox, state)

  [Online Generation Loop]
  5. ObjectRegistry.to_padded_tensor() → obj_state tensor 구성
  6. forward_wm() — action + text + obj_state token conditioning으로 프레임 chunk 생성
  7. 생성된 각 프레임에 대해:
     a. SAM3Manager.update() → mask 전파
     b. interaction 판별 (로봇팔 mask와 물체 mask의 IoU 기준)
     c. ObjectRegistry 갱신 (presence, appearance, bbox, state)
  8. Negative detection:
     - out_of_frame / occluded / vanished 이벤트 검출
     - 이벤트 발생 시: last_good_frame/mask 기준으로 mask·identity 수정
     - 필요시 rollback (last_good 시점으로 his_cond 되돌리기)
  9. 수정된 conditioning으로 다음 chunk 생성 반복
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import datetime
import numpy as np
import torch
import einops
import mediapy
from decord import VideoReader, cpu

from config import wm_args
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.object_registry import ObjectRegistry, FEATURE_DIM, MAX_OBJECTS
from sam3_manager import SAM3Manager
from accelerate import Accelerator
from argparse import ArgumentParser


# ─────────────────────────────────────────────────────────────────
# 유틸: 첫 프레임 로드 (모델 없이, rollout_key_board.py의 get_traj_info와 동일 로직)
# ─────────────────────────────────────────────────────────────────
def get_frames(args, traj_id, start_idx=0, steps=8):
    val_dataset_dir = args.val_dataset_dir
    skip = args.skip_step
    annotation_path = f"{val_dataset_dir}/annotation/val/{traj_id}.json"
    with open(annotation_path) as f:
        anno = json.load(f)
    try:
        length = len(anno['action'])
    except:
        length = anno["video_length"]

    frames_ids = np.arange(start_idx, start_idx + steps * skip, skip)
    max_ids = np.ones_like(frames_ids) * (length - 1)
    frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)

    instruction = anno['texts'][0]
    car_action = np.array(anno['states'])[frames_ids]
    joint_pos = np.array(anno['joints'])[frames_ids]

    video_dict = []
    for view_info in anno['videos']:
        video_path = f"{val_dataset_dir}/{view_info['video_path']}"
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        try:
            frames = vr.get_batch(range(length)).asnumpy()
        except:
            frames = vr.get_batch(range(length)).numpy()
        video_dict.append(frames[frames_ids])

    return car_action, joint_pos, video_dict, anno, instruction


# ─────────────────────────────────────────────────────────────────
# Qwen3-VL: 장면 설명 → 객체 목록 + SAM3 텍스트 프롬프트 생성
# ─────────────────────────────────────────────────────────────────
def build_object_prompts_with_qwen(frame: np.ndarray, instruction: str, qwen_model, qwen_processor, device: str) -> list[str]:
    """
    첫 프레임 + task instruction을 Qwen3-VL에 넣어
    장면에 있는 객체 목록을 텍스트로 받아 SAM3 프롬프트 리스트로 반환.

    반환 예시: ["robot arm and end-effector", "red cup", "table"]
    """
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
        {"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": prompt_text},
        ]}
    ]
    text_input = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text_input], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = qwen_model.generate(**inputs, max_new_tokens=128)
    response = qwen_processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # 줄 단위로 파싱
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    # robot arm이 없으면 강제로 추가
    if not any('robot arm' in l.lower() for l in lines):
        lines = ['robot arm and end-effector'] + lines
    return lines


# ─────────────────────────────────────────────────────────────────
# interaction 판별: 로봇팔 mask와 물체 mask의 IoU
# ─────────────────────────────────────────────────────────────────
def is_interaction(robot_mask: np.ndarray, obj_mask: np.ndarray, iou_thresh: float = 0.05) -> bool:
    if robot_mask is None or obj_mask is None:
        return False
    intersection = (robot_mask & obj_mask).sum()
    union = (robot_mask | obj_mask).sum()
    if union == 0:
        return False
    return (intersection / union) > iou_thresh


# ─────────────────────────────────────────────────────────────────
# ObjectRegistry 갱신 (매 프레임)
# ─────────────────────────────────────────────────────────────────
def update_registry(
    registry: ObjectRegistry,
    sam_results: dict,
    frame: np.ndarray,
    robot_label: str,
    clip_model,
    clip_processor,
    device: str,
):
    """
    SAM3Manager.update() 결과를 받아 ObjectRegistry를 갱신.
    - interaction 상태 판별
    - absence 이벤트 처리
    """
    robot_mask = sam_results.get(robot_label, {}).get('mask')

    for label, result in sam_results.items():
        mask = result['mask']
        absent = result['absent']
        cause = result['cause']

        if absent or mask is None:
            registry.mark_absent(label)
            continue

        # appearance embedding (CLIP)
        if clip_model is not None:
            appearance = registry.extract_appearance(frame, mask, clip_model, clip_processor, device)
        else:
            appearance = np.zeros(512, dtype=np.float32)

        bbox = ObjectRegistry.mask_to_bbox(mask, frame.shape[:2])

        # interaction 상태 판별 (로봇팔 제외한 객체만)
        if label != robot_label and is_interaction(robot_mask, mask):
            state = 1.0   # interaction
        else:
            state = 0.0   # normal

        registry.update(
            label=label,
            presence=1.0,
            appearance=appearance,
            bbox=bbox,
            state=state,
            frame=frame,
            mask=mask,
        )


# ─────────────────────────────────────────────────────────────────
# 메인 Agent: rollout_key_board.py의 agent를 상속해서 obj_state 지원 추가
# ─────────────────────────────────────────────────────────────────
class OnlineAgent:
    def __init__(self, args):
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        self.model = CrtlWorld(args)
        # strict=False: 기존 ckpt에 object_state_encoder weight가 없어도 로드 가능
        # (object_state_encoder는 zero init이므로 weight 없어도 zeros로 시작)
        missing, unexpected = self.model.load_state_dict(
            torch.load(args.val_model_path, map_location='cpu'), strict=False
        )
        if missing:
            print(f"[WARN] missing keys (new modules): {missing}")
        if unexpected:
            print(f"[WARN] unexpected keys: {unexpected}")
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        print("load world model success")

        with open(args.data_stat_path, 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None, :]
            self.state_p99 = np.array(data_stat['state_99'])[None, :]

    def normalize_bound(self, data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def forward_wm(self, action_cond, video_latent_cond, his_cond, obj_state_np, text):
        """
        action_cond:      (T, 7) np.ndarray
        video_latent_cond:(1, 4, 72, 40) torch.Tensor
        his_cond:         (1, num_history, 4, 72, 40) torch.Tensor
        obj_state_np:     (MAX_OBJECTS, FEATURE_DIM) np.ndarray
        text:             str
        returns: (latents, decoded_frames)
            latents:        (3, num_frames, 4, h, w) torch.Tensor
            decoded_frames: list of np.ndarray (H, W, 3) uint8, length=num_frames
        """
        args = self.args
        pipeline = self.model.pipeline

        # normalize action
        action_normed = self.normalize_bound(action_cond, self.state_p01, self.state_p99)
        action_t = torch.tensor(action_normed).unsqueeze(0).to(self.device).to(self.dtype)

        with torch.no_grad():
            # build action+text token (action_encoder)
            text_token = self.model.action_encoder(
                action_t, [text], self.model.tokenizer, self.model.text_encoder
            )  # (1, T, 1024)

            # obj_token은 fine-tune 완료 후에만 inject
            # fine-tune 전 checkpoint는 T==num_frames(1 token/frame)만으로 학습됐으므로
            # zero obj_token을 concat하면 cross-attention 분포가 깨져 생성 품질 저하
            obj_state_nonzero = (obj_state_np.sum() != 0) and self.args.use_obj_token
            if obj_state_nonzero:
                obj_t = torch.tensor(obj_state_np, dtype=self.dtype, device=self.device).unsqueeze(0)
                obj_tokens = self.model.object_state_encoder(obj_t)  # (1, N, 1024)
                encoder_hidden = torch.cat([text_token, obj_tokens], dim=1)  # (1, T+N, 1024)
            else:
                encoder_hidden = text_token  # (1, T, 1024) — 기존 경로 그대로

        # diffusion sampling
        with torch.no_grad():
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=video_latent_cond,
                text=encoder_hidden,
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
            )
        # (1, num_frames, 4, 3H, W) → (3, num_frames, 4, H, W)
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)

        # decode predicted frames
        decoded_frames = []
        bsz, frame_num = latents.shape[:2]
        x = latents.flatten(0, 1)
        decode_kwargs = {}
        decoded = []
        for i in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[i:i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded, dim=0)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        # view 1 (wrist cam) 기준으로 프레임 리스트 반환
        view_idx = 1
        decoded_frames = [videos[view_idx, t] for t in range(frame_num)]

        return latents, decoded_frames


# ─────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path',        type=str, default=None)
    parser.add_argument('--svd_model_path',   type=str, default=None)
    parser.add_argument('--clip_model_path',  type=str, default=None)
    parser.add_argument('--task_type',        type=str, default='keyboard')
    parser.add_argument('--keyboard',         type=str, default='ddcu')
    parser.add_argument('--sam3_ckpt',        type=str, default='/home/dgu/minyoung/sam3/checkpoints/sam3.1_multiplex.pt')
    parser.add_argument('--qwen_model_path',  type=str, default='/home/dgu/minyoung/models/Qwen3-VL-8B-Instruct',
                        help='Qwen3-VL 모델 경로. None이면 장면 객체를 직접 지정해야 함')
    parser.add_argument('--object_labels',    type=str, default=None,
                        help='Qwen3 없을 때 수동 지정. 예: "robot arm and end-effector,cup"')
    parser.add_argument('--view_idx',         type=int, default=1,
                        help='SAM3 추적에 사용할 카메라 뷰 인덱스')
    parser.add_argument('--iou_interaction',  type=float, default=0.05)
    parser.add_argument('--rollback_on_neg',  action='store_true',
                        help='Negative event 발생 시 rollback 수행 여부')
    parser.add_argument('--use_obj_token',    action='store_true',
                        help='obj_state token을 UNet에 주입 (fine-tune 완료 후에만 사용)')
    parser.add_argument('--debug_dir',        type=str, default=None,
                        help='SAM3 시각화(mask+box) 저장 디렉터리. 미지정 시 저장 안 함')
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)
    for k, v in vars(args_new).items():
        if v is not None:
            setattr(args, k, v)

    # ── 모델 로드 ────────────────────────────────────────────
    agent = OnlineAgent(args)
    VIEW_IDX = args_new.view_idx

    # ── SAM3 ────────────────────────────────────────────────
    sam_manager = SAM3Manager(checkpoint_path=args_new.sam3_ckpt, device=str(agent.device))

    # ── Qwen3-VL (optional) ──────────────────────────────────
    clip_model, clip_processor = None, None
    qwen_model, qwen_processor = None, None
    if args_new.qwen_model_path and os.path.exists(args_new.qwen_model_path):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print(f"Loading Qwen3-VL from {args_new.qwen_model_path} ...")
        qwen_model = AutoModelForVision2Seq.from_pretrained(
            args_new.qwen_model_path, torch_dtype=torch.bfloat16
        ).to(agent.device)
        qwen_processor = AutoProcessor.from_pretrained(args_new.qwen_model_path)

        # appearance embedding용 CLIP image encoder
        from transformers import CLIPVisionModelWithProjection, CLIPProcessor
        clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model_path).to(agent.device)
        clip_model.eval()
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model_path)

    # ── rollout ──────────────────────────────────────────────
    interact_num = len(args.keyboard)
    pred_step    = args.pred_step
    num_history  = args.num_history
    num_frames   = args.num_frames

    for val_id_i, start_idx_i in zip(args.val_id, args.start_idx):
        eef_gt, joint_pos_gt, video_dict, anno, instruction = get_frames(
            args, val_id_i, start_idx=start_idx_i, steps=int(pred_step * interact_num + 8)
        )
        print(f"traj={val_id_i}, instruction={instruction}")

        # ── [Initialization] ─────────────────────────────────
        first_frame = video_dict[VIEW_IDX][0]  # (H, W, 3) uint8

        # Step 2: 객체 목록 결정
        if qwen_model is not None:
            object_labels = build_object_prompts_with_qwen(
                first_frame, instruction, qwen_model, qwen_processor, str(agent.device)
            )
        elif args_new.object_labels:
            object_labels = [l.strip() for l in args_new.object_labels.split(',')]
        else:
            object_labels = ['robot arm and end-effector']
        print(f"Object labels: {object_labels}")

        # Step 3: SAM3 초기화
        sam_manager.initialize(first_frame, object_labels)

        # Step 4: ObjectRegistry 초기화
        registry = ObjectRegistry()
        for label in object_labels:
            registry.register(label)

        robot_label = object_labels[0]  # 첫 번째가 항상 robot arm
        for label in object_labels:
            mask = sam_manager.object_masks.get(label)
            if mask is not None:
                appearance = (registry.extract_appearance(first_frame, mask, clip_model, clip_processor, str(agent.device))
                              if clip_model is not None else np.zeros(512, dtype=np.float32))
                bbox = ObjectRegistry.mask_to_bbox(mask, first_frame.shape[:2])
                registry.update(label, presence=1.0, appearance=appearance, bbox=bbox, state=0.0,
                                 frame=first_frame, mask=mask)
            else:
                registry.mark_absent(label)

        # VAE로 latent 인코딩 (첫 프레임)
        vae = agent.model.pipeline.vae
        def encode_frame(frame_np):
            x = torch.from_numpy(frame_np).to(agent.dtype).to(agent.device)
            x = x.permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
            return latent  # (1, 4, H, W)

        # 모든 뷰 latent를 concat → (1, 4, 72, 40)
        def encode_all_views(frame_list):
            latents = [encode_frame(f) for f in frame_list]
            return torch.cat(latents, dim=2)  # (1, 4, 72, 40)

        first_latent = encode_all_views([video_dict[v][0] for v in range(len(video_dict))])  # (1, 4, 72, 40)

        # history buffer 초기화
        his_cond  = [first_latent] * (num_history * 4)
        his_eef   = [eef_gt[0:1]] * (num_history * 4)

        video_to_save = []
        vis_frames_to_save = []   # SAM3 mask overlay 확인용
        rollback_count = 0

        # ── [Online Generation Loop] ─────────────────────────
        for i in range(interact_num):
            print(f"\n── Step {i+1}/{interact_num} ──")

            # action 준비 (keyboard control)
            from models.utils import key_board_control
            history_idx = [0, 0, -8, -6, -4, -2]
            his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
            current_pose = his_eef[-1]
            cartesian_pose = key_board_control(current_pose, args.keyboard[i], task_id=val_id_i)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)

            # his_cond 입력 구성
            his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]

            # Step 5: obj_state tensor
            obj_state_np = registry.to_padded_tensor(MAX_OBJECTS)  # (MAX_OBJECTS, FEATURE_DIM)

            # Step 6: world model forward
            pred_latents, pred_frames = agent.forward_wm(
                action_cond, current_latent, his_cond_input, obj_state_np, instruction
            )

            # Step 7: 생성된 프레임 분석 (3장 단위 chunk, 프레임별 absence 판정)
            neg_event_detected = False
            first_bad_t = None          # chunk 내 첫 bad frame 인덱스

            SAM3_CHUNK = 3         # chunk 크기 고정
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                      (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]

            # pred_frames 를 SAM3_CHUNK 단위로 잘라서 처리
            sam_results_flat = []
            for chunk_start in range(0, len(pred_frames), SAM3_CHUNK):
                chunk = pred_frames[chunk_start: chunk_start + SAM3_CHUNK]
                chunk_results = sam_manager.update_chunk(chunk)
                sam_results_flat.extend(chunk_results)
                # 이 chunk 안에서 이미 bad frame 발생 → 나머지 chunk 처리 중단
                if any(
                    any(r.get("first_bad_t") is not None for r in frame_res.values())
                    for frame_res in chunk_results
                ):
                    break

            for t, (frame, sam_results) in enumerate(zip(pred_frames, sam_results_flat)):
                # ObjectRegistry 갱신
                update_registry(registry, sam_results, frame, robot_label,
                                 clip_model, clip_processor, str(agent.device))

                # SAM3 mask overlay 시각화
                vis_frame = frame.copy()
                for ci, (label, result) in enumerate(sam_results.items()):
                    mask = result.get('mask')
                    if mask is None:
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
                        if result.get('absent'):
                            vis_frame[y1:y2, x1:x2] = (
                                vis_frame[y1:y2, x1:x2] * 0.5 + np.array([255, 0, 0]) * 0.5
                            ).astype(np.uint8)
                vis_frames_to_save.append(vis_frame)

                # Step 8: Negative detection (프레임별)
                for label, result in sam_results.items():
                    if result.get('absent'):
                        cause = result['cause']
                        fbt = result.get('first_bad_t')
                        print(f"  [NEG] '{label}' absent: {cause} at chunk-frame {t} (first_bad_t={fbt})")
                        neg_event_detected = True
                        # first_bad_t는 chunk-local 인덱스 → 전체 pred_frames 인덱스로 변환
                        chunk_offset = (t // SAM3_CHUNK) * SAM3_CHUNK
                        first_bad_t = chunk_offset + (fbt if fbt is not None else (t % SAM3_CHUNK))
                        break

                if neg_event_detected:
                    break

            # Rollback 처리: first_bad_t 이후만 버림 (이전 프레임은 유지)
            if neg_event_detected and args_new.rollback_on_neg:
                print(f"  [ROLLBACK] step={i}, first_bad_t={first_bad_t}")
                rollback_count += 1

                for label in object_labels:
                    registry.rollback(label)

                # first_bad_t 이전 프레임만 저장 (0이면 전체 버림)
                pred_frames = pred_frames[:first_bad_t] if first_bad_t else []

            # history buffer 업데이트
            his_eef.append(cartesian_pose[pred_step - 1:pred_step])
            his_cond.append(torch.cat([pred_latents[v][pred_step - 1] for v in range(3)], dim=1).unsqueeze(0))

            # 저장할 프레임 누적
            if i == interact_num - 1:
                video_to_save.extend(pred_frames)
            else:
                video_to_save.extend(pred_frames[:pred_step - 1])

        # ── 저장 ─────────────────────────────────────────────
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"{args.save_dir}/{args.task_name}/video"
        os.makedirs(base_dir, exist_ok=True)

        # 생성 영상
        video_arr = np.stack(video_to_save, axis=0)
        save_path = f"{base_dir}/online_{uuid}_{val_id_i}_{start_idx_i}_{args.keyboard}.mp4"
        mediapy.write_video(save_path, video_arr, fps=4)
        print(f"\nSaved: {save_path}  (rollbacks={rollback_count})")

        # SAM3 mask overlay 영상 (tracking 확인용)
        if vis_frames_to_save:
            vis_arr = np.stack(vis_frames_to_save, axis=0)
            vis_path = f"{base_dir}/online_{uuid}_{val_id_i}_{start_idx_i}_{args.keyboard}_sam3.mp4"
            mediapy.write_video(vis_path, vis_arr, fps=4)
            print(f"SAM3 overlay: {vis_path}")
