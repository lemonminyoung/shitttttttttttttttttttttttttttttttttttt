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

import random #seed 고정하려고 놓은 거임

from config import wm_args
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.object_registry import ObjectRegistry, FEATURE_DIM, MAX_OBJECTS
from sam3_manager_new import SAM3ManagerNew as SAM3Manager
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
    iou_thresh: float = 0.05,
    initial_areas: dict = None,
):
    """
    SAM3Manager.update() 결과를 받아 ObjectRegistry를 갱신.
    - interaction 상태 판별
    - absence 이벤트 처리
    returns: {label: {"iou": float | None, "state": float}}
    """
    print(f"[IOU THRESH] {iou_thresh}")
    robot_mask = sam_results.get(robot_label, {}).get('mask')
    interact_info: dict = {}

    for label, result in sam_results.items():
        mask = result['mask']
        absent = result['absent']
        cause = result['cause']

        if absent or mask is None: #
            registry.mark_absent(label)
            interact_info[label] = {"iou": None, "state": -1.0}
            continue

        # appearance embedding (CLIP)
        if clip_model is not None:
            appearance = registry.extract_appearance(frame, mask, clip_model, clip_processor, device)
        else:
            appearance = np.zeros(512, dtype=np.float32)

        bbox = ObjectRegistry.mask_to_bbox(mask, frame.shape[:2])

        # interaction 상태 판별 (로봇팔 제외한 객체만)
        if label != robot_label and robot_mask is not None and mask is not None:
            intersection = float((robot_mask & mask).sum())
            # 초기 면적 기반 적응형 metric: intersection / initial_area
            # 작은 객체(펜 등)도 초기 크기 대비 overlap 비율로 균등 판정
            init_area = (initial_areas or {}).get(label, 0.0) if initial_areas else 0.0
            if init_area > 0:
                metric = intersection / init_area
            else:
                union = float((robot_mask | mask).sum())
                metric = intersection / union if union > 0 else 0.0
            state = 1.0 if metric > iou_thresh else 0.0
            print(f"[INTERACT] label={label} iou={metric:.4f} thresh={iou_thresh} state={state}")
        else:
            metric = None
            state = 0.0

        interact_info[label] = {"iou": round(metric, 6) if metric is not None else None, "state": state}

        registry.update(
            label=label,
            presence=1.0,
            appearance=appearance,
            bbox=bbox,
            state=state,
            frame=frame,
            mask=mask,
        )

    return interact_info


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 가능한 범위에서 재현성 강화
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.x 권장 옵션
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # cuBLAS 재현성
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

        # 3개 view를 가로로 concat해서 반환 (view0 | view1 | view2)
        # SAM3 tracking은 view_idx(기본 1) 기준으로 별도 추출
        view_idx = getattr(self.args, 'view_idx', 1)
        decoded_frames_track = [videos[view_idx, t] for t in range(frame_num)]
        decoded_frames_full  = [
            np.concatenate([videos[0, t], videos[1, t], videos[2, t]], axis=1)
            for t in range(frame_num)
        ]

        return latents, decoded_frames_track, decoded_frames_full


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
    parser.add_argument('--sam3_ckpt',        type=str, default='/home/dgu/minyoung/sam3/checkpoints/sam3.pt')
    parser.add_argument('--qwen_model_path',  type=str, default='/home/dgu/minyoung/models/Qwen3-VL-8B-Instruct',
                        help='Qwen3-VL 모델 경로. None이면 장면 객체를 직접 지정해야 함')
    parser.add_argument('--object_labels',    type=str, default=None,
                        help='Qwen3 없을 때 수동 지정. 예: "robot arm and end-effector,cup"')
    parser.add_argument('--view_idx',         type=int, default=1,
                        help='SAM3 추적에 사용할 카메라 뷰 인덱스')
    parser.add_argument('--iou_interaction',  type=float, default=0.05,
                        help='로봇팔-물체 interaction 판별 IoU threshold')
    parser.add_argument('--rollback_on_neg',       action='store_true',
                        help='Negative event 발생 시 rollback+retry 수행 여부')
    parser.add_argument('--max_retries',           type=int, default=3,
                        help='스텝별 최대 rollback retry 횟수')
    parser.add_argument('--rollback_trim_margin',  type=int, default=1,
                        help='rollback 시 first_bad_t에서 몇 프레임 앞까지 잘라낼지 (기본 1)')
    parser.add_argument('--use_obj_token',    action='store_true',
                        help='obj_state token을 UNet에 주입 (fine-tune 완료 후에만 사용)')
    parser.add_argument('--seed', type=int, default=42, #########
                        help='random seed for reproducible rollout generation')
    parser.add_argument('--raw_only', action='store_true',
                    help='save plain Ctrl-World video only, skip SAM checks/logs')
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)
    for k, v in vars(args_new).items():
        if v is not None:
            setattr(args, k, v)

    print(f"[SEED] {args.seed}")
    set_seed(args.seed)

    # ── 모델 로드 ────────────────────────────────────────────
    agent = OnlineAgent(args)
    VIEW_IDX = args_new.view_idx

    # ── SAM3 ────────────────────────────────────────────────
    sam_manager = SAM3Manager(checkpoint_path=args_new.sam3_ckpt, device=str(agent.device))



    # ── CLIP (appearance 비교용 — Qwen 여부와 무관하게 항상 로드) ──────
    clip_model, clip_processor = None, None
    if args.clip_model_path and os.path.exists(str(args.clip_model_path)):
        from transformers import CLIPVisionModelWithProjection, CLIPProcessor
        print(f"Loading CLIP from {args.clip_model_path} ...")
        clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model_path).to(agent.device)
        clip_model.eval()
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    else:
        print("[WARN] clip_model_path not found — appearance similarity disabled")

    # ── Qwen3-VL (optional) ──────────────────────────────────
    qwen_model, qwen_processor = None, None
    if args_new.qwen_model_path and os.path.exists(args_new.qwen_model_path):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print(f"Loading Qwen3-VL from {args_new.qwen_model_path} ...")
        qwen_model = AutoModelForVision2Seq.from_pretrained(
            args_new.qwen_model_path, torch_dtype=torch.bfloat16
        ).to(agent.device)
        qwen_processor = AutoProcessor.from_pretrained(args_new.qwen_model_path)

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
        if qwen_model is not None: # and mask.sum() > 0:
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

        # 객체별 초기 appearance 저장 (crushed 판별용 cosine similarity 기준점)
        initial_appearances: dict = {}
        for label in object_labels:
            mask = sam_manager.object_masks.get(label)
            if mask is not None:
                app = (registry.extract_appearance(first_frame, mask, clip_model, clip_processor, str(agent.device))
                       if clip_model is not None else np.zeros(512, dtype=np.float32))
                initial_appearances[label] = app
                bbox = ObjectRegistry.mask_to_bbox(mask, first_frame.shape[:2])
                registry.update(label, presence=1.0, appearance=app, bbox=bbox, state=0.0,
                                 frame=first_frame, mask=mask)
            else:
                initial_appearances[label] = None
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
        vis_frames_to_save = []
        tracking_log  = {label: [] for label in object_labels}
        frame_counter = 0
        rollback_count = 0
        false_gen_count = 0

        # uuid / base_dir — 루프 전에 확정 (false_gen 파일명에도 사용)
        uuid     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"{args.save_dir}/{args.task_name}/video"
        os.makedirs(base_dir, exist_ok=True)

        from models.utils import key_board_control

        SAM3_CHUNK          = 3
        REDETECT_SIM_THRESH = 0.70
        TRIM_MARGIN         = args_new.rollback_trim_margin   # first_bad_t 앞 몇 프레임 버릴지
        MAX_RETRIES         = args_new.max_retries
        colors = [(0, 255, 0), (0, 0, 255), (0, 150, 150),
                  (0, 150, 255), (0, 255, 150), (0, 128, 200), (0, 128, 0)]

        # ── 에러 심각도 / rollback threshold ────────────────────────────
        CAUSE_SEVERITY = {
            'occluded':      0.5,   # 잠깐 가려진 것 — 가장 가벼움
            'out_of_frame':  0.7,   # 화면 밖
            'vanished':      1.0,   # 사라짐
            'crushed':       1.5,   # mask 폭발 — 가장 무거움
        }
        # Tier 1 soft recovery 최대 연속 허용 프레임 (초과 시 vanished 승격)
        SOFT_RECOVER_MAX = 3
        # 일반 객체: streak 2프레임 이상 or 누적 score >= 2.0 → rollback
        DEFAULT_STREAK_THRESH = 2
        DEFAULT_SCORE_THRESH  = 2.0
        # robot arm 전용: 더 보수적 (단독 문제로 전체 rollback 방지)
        ROBOT_STREAK_THRESH   = 4
        ROBOT_SCORE_THRESH    = 3.5

        # ── 에러 점수: step 간 retry에서 감쇠 유지 (while 루프 전에 선언) ──
        bad_streak  = {label: 0   for label in object_labels}
        error_score = {label: 0.0 for label in object_labels}
        soft_streak = {label: 0   for label in object_labels}  # Tier 1 연속 횟수

        # ── [Online Generation Loop — while 기반 retry 구조] ─────────
        i = 0
        step_retry_count = 0

        while i < interact_num:
            print(f"\n── Step {i+1}/{interact_num}  retry={step_retry_count} ──")

            # ── 스텝 시작 전 스냅샷 ─────────────────────────────────
            his_cond_snap      = list(his_cond)
            his_eef_snap       = list(his_eef)
            frame_counter_snap = frame_counter
            registry_snap      = registry.snapshot()

            # ── 스텝별 임시 버퍼 (확정 후 main buffer에 병합) ────────
            step_full = []   # pred_frames_full (3-view concat)
            step_vis  = []   # SAM3 overlay
            step_log  = {label: [] for label in object_labels}

            # 에러 점수 — retry 시 감쇠 유지, 새 생성마다 soft_streak만 리셋
            if step_retry_count > 0:
                for label in object_labels:
                    bad_streak[label]  = int(bad_streak[label]  * 0.5)
                    error_score[label] = round(error_score[label] * 0.5, 3)
            soft_streak = {label: 0 for label in object_labels}  # 새 프레임 생성마다 리셋

            # ── Action 준비 ─────────────────────────────────────────
            history_idx  = [0, 0, -8, -6, -4, -2]
            his_pose     = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
            current_pose = his_eef[-1]
            cartesian_pose = key_board_control(current_pose, args.keyboard[i], task_id=val_id_i)
            action_cond    = np.concatenate([his_pose, cartesian_pose], axis=0)

            his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]
            obj_state_np   = registry.to_padded_tensor(MAX_OBJECTS)

            # ── World Model Forward ─────────────────────────────────
            pred_latents, pred_frames_track, pred_frames_full = agent.forward_wm(
                action_cond, current_latent, his_cond_input, obj_state_np, instruction
            )

            if args_new.raw_only:
                if i == interact_num - 1:
                    video_to_save.extend(pred_frames_full)
                else:
                    video_to_save.extend(pred_frames_full[:pred_step - 1])

                his_eef.append(cartesian_pose[pred_step - 1:pred_step])
                his_cond.append(torch.cat([pred_latents[v][pred_step - 1] for v in range(3)], dim=1).unsqueeze(0))
                i += 1
                continue

            # ── SAM3 Chunk Tracking ─────────────────────────────────
            sam_results_flat = []
            for chunk_start in range(0, len(pred_frames_track), SAM3_CHUNK):
                chunk = pred_frames_track[chunk_start: chunk_start + SAM3_CHUNK]
                chunk_results = sam_manager.update_chunk(chunk)
                sam_results_flat.extend(chunk_results)
                if any(
                    any(r.get("first_bad_t") is not None for r in fr.values())
                    for fr in chunk_results
                ):
                    break

            # ── 프레임별 처리 ────────────────────────────────────────
            neg_event_detected = False
            first_bad_t = None

            for t, (frame_track, frame_full, sam_results) in enumerate(
                zip(pred_frames_track, pred_frames_full, sam_results_flat)
            ):
                # ── 3-tier recovery + 에러 점수 누적 ─────────────────────
                redetected_labels:     set  = set()
                soft_recovered_labels: set  = set()
                recovery_tier:         dict = {}   # {label: 0=정상 1=soft 2=redetect 3=hard}
                rollback_candidate:    dict = {}   # {label: bool}

                for label in object_labels:
                    result   = sam_results.get(label, {})
                    is_robot = (label == robot_label)

                    # robot arm: crushed 판단 무시 (mask 폭발을 생성 오류로 취급하지 않음)
                    if is_robot and result.get('cause') == 'crushed':
                        sam_results[label]['cause']  = None
                        sam_results[label]['absent'] = False
                        result = sam_results[label]

                    if not (result.get('absent') or result.get('mask') is None):
                        # 정상 프레임: streak 초기화, score 감쇠
                        bad_streak[label]  = 0
                        soft_streak[label] = 0
                        error_score[label] = max(0.0, error_score[label] - 0.3)
                        recovery_tier[label]      = 0
                        rollback_candidate[label] = False
                        continue

                    cause = result.get('cause') or 'vanished'

                    # ── Tier 1: Soft recovery (occluded, ≤ SOFT_RECOVER_MAX frames) ──
                    if cause == 'occluded' and soft_streak[label] < SOFT_RECOVER_MAX:
                        obj = registry.get(label)
                        soft_mask = obj.last_good_mask.copy() if obj.last_good_mask is not None else None
                        if soft_mask is not None:
                            # 필드 일관성: mask/absent/cause 모두 정상 상태로 통일
                            sam_results[label]['mask']   = soft_mask
                            sam_results[label]['absent'] = False
                            sam_results[label]['cause']  = None
                            soft_recovered_labels.add(label)
                        soft_streak[label] += 1
                        bad_streak[label]  += 1
                        error_score[label] += CAUSE_SEVERITY.get('occluded', 0.5)
                        recovery_tier[label] = 1
                        streak_thresh = ROBOT_STREAK_THRESH if is_robot else DEFAULT_STREAK_THRESH
                        score_thresh  = ROBOT_SCORE_THRESH  if is_robot else DEFAULT_SCORE_THRESH
                        rollback_candidate[label] = (
                            bad_streak[label] >= streak_thresh or error_score[label] >= score_thresh
                        )
                        print(f"  [SOFT] '{label}': occluded  soft_streak={soft_streak[label]}/{SOFT_RECOVER_MAX}"
                              f"  streak={bad_streak[label]}  score={error_score[label]:.2f}"
                              f"  candidate={rollback_candidate[label]}")
                        continue

                    # soft_streak 한도 초과: occluded → vanished로 승격
                    if cause == 'occluded':
                        cause = 'vanished'
                        sam_results[label]['cause'] = 'vanished'
                        print(f"  [ESCALATE] '{label}': soft_streak={soft_streak[label]} 초과 → vanished 승격")
                    soft_streak[label] = 0  # soft chain 종료

                    # ── Tier 2: Re-detection ─────────────────────────────────
                    # crushed(비-로봇) → 생성 오류로 간주, redetect 없이 바로 Tier 3
                    skip_redetect = (cause == 'crushed' and not is_robot)

                    if not skip_redetect:
                        print(f"  [REDETECT] '{label}': tracking lost (cause={cause}), re-detecting...")
                        redet_mask, _redet_box = sam_manager.redetect(frame_track, label)
                        if redet_mask is not None:
                            resumed = True
                            init_emb = initial_appearances.get(label)
                            if clip_model is not None and init_emb is not None and init_emb.sum() != 0:
                                curr_emb = registry.extract_appearance(
                                    frame_track, redet_mask, clip_model, clip_processor, str(agent.device)
                                )
                                sim = float(np.dot(curr_emb, init_emb) /
                                            (np.linalg.norm(curr_emb) * np.linalg.norm(init_emb) + 1e-8))
                                print(f"  [REDETECT] '{label}': cosine_sim={sim:.4f}"
                                      f"  diff={1-sim:.4f}  thresh={REDETECT_SIM_THRESH}")
                                resumed = sim >= REDETECT_SIM_THRESH
                                if not resumed:
                                    print(f"  [REDETECT] '{label}': appearance mismatch → vanished")
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

                    # ── Tier 3: hard failure — 에러 누적 ────────────────────
                    severity = CAUSE_SEVERITY.get(cause, 1.0)
                    bad_streak[label]  += 1
                    error_score[label] += severity
                    sam_results[label]['absent'] = True
                    recovery_tier[label] = 3

                    streak_thresh = ROBOT_STREAK_THRESH if is_robot else DEFAULT_STREAK_THRESH
                    score_thresh  = ROBOT_SCORE_THRESH  if is_robot else DEFAULT_SCORE_THRESH
                    rollback_candidate[label] = (
                        bad_streak[label] >= streak_thresh or error_score[label] >= score_thresh
                    )
                    print(f"  [SCORE] '{label}': cause={cause}"
                          f"  streak={bad_streak[label]}  score={error_score[label]:.2f}"
                          f"  thresh(streak={streak_thresh}, score={score_thresh:.1f})"
                          f"  candidate={rollback_candidate[label]}")

                # ── 단일 rollback 판단 포인트 ───────────────────────────────
                # robot arm 단독 문제는 rollback 금지
                # non-robot 객체 중 하나라도 rollback_candidate → rollback
                non_robot_trigger = [
                    lbl for lbl in object_labels
                    if lbl != robot_label and rollback_candidate.get(lbl, False)
                ]
                if non_robot_trigger and not neg_event_detected:
                    neg_event_detected = True
                    first_bad_t = min(
                        max(0, t - bad_streak[lbl] + 1) for lbl in non_robot_trigger
                    )
                    print(f"  [NEG] ROLLBACK triggered by {non_robot_trigger}"
                          f"  first_bad_t={first_bad_t}")

                # ObjectRegistry 갱신
                interact_info = update_registry(
                    registry, sam_results, frame_track, robot_label,
                    clip_model, clip_processor, str(agent.device),
                    iou_thresh=args_new.iou_interaction,
                    initial_areas=sam_manager.initial_areas,
                )

                # 스텝 tracking log (최종 후처리 판정값 기준)
                for label in object_labels:
                    result = sam_results.get(label, {})
                    mask   = result.get('mask')
                    area   = float(mask.sum()) if mask is not None else 0.0
                    bbox   = None
                    if mask is not None:
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    appearance_sim = None
                    if clip_model is not None and mask is not None:
                        curr_emb = registry.extract_appearance(
                            frame_track, mask, clip_model, clip_processor, str(agent.device)
                        )
                        init_emb = initial_appearances.get(label)
                        if init_emb is not None and init_emb.sum() != 0:
                            sim = float(np.dot(curr_emb, init_emb) /
                                        (np.linalg.norm(curr_emb) * np.linalg.norm(init_emb) + 1e-8))
                            appearance_sim = round(sim, 4)
                            if result.get('cause') == 'crushed':
                                print(f"  [APPEARANCE] '{label}' crushed: cosine_sim={sim:.4f}  diff={1-sim:.4f}")
                    info = interact_info.get(label, {})
                    step_log[label].append({
                        "frame":              frame_counter_snap + t,
                        "area":               area,
                        "bbox":               bbox,
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
                    })

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
                        final_absent = result.get('absent') or (result.get('mask') is None) or (result.get('cause') in {'vanished', 'crushed'})
                        if final_absent:
                        #if result.get('absent'):
                            vis_frame[y1:y2, x1:x2] = (
                                vis_frame[y1:y2, x1:x2] * 0.5 + np.array([255, 0, 0]) * 0.5
                            ).astype(np.uint8)

                step_full.append(frame_full)
                step_vis.append(vis_frame)

                if neg_event_detected:
                    break

            # ── 롤백 vs 진행 결정 ───────────────────────────────────
            do_rollback = (neg_event_detected
                           and args_new.rollback_on_neg
                           and step_retry_count < MAX_RETRIES)

            if do_rollback:
                trim_t = max(0, (first_bad_t or 0) - TRIM_MARGIN)
                print(f"[ROLLBACK] step={i}  retry={step_retry_count+1}/{MAX_RETRIES}"
                      f"  first_bad_t={first_bad_t}  trim_t={trim_t}")

                # false_generation 파일 저장 (first_bad_t 이후 bad 프레임들)
                fg_start  = first_bad_t if first_bad_t is not None else trim_t
                fg_frames = step_full[fg_start:]
                if fg_frames:
                    fg_arr  = np.stack(fg_frames, axis=0)
                    fg_path = (f"{base_dir}/false_gen_{uuid}_{val_id_i}_{start_idx_i}"
                               f"_{args.keyboard}_step{i}_r{step_retry_count}.mp4")
                    mediapy.write_video(fg_path, fg_arr, fps=4)
                    print(f"[FALSE_GEN] {fg_path}  ({len(fg_frames)} frames, start_t={fg_start})")
                    false_gen_count += 1

                # 안전 프레임만 main buffer에 반영
                video_to_save.extend(step_full[:trim_t])
                vis_frames_to_save.extend(step_vis[:trim_t])
                for label in object_labels:
                    tracking_log[label].extend(step_log[label][:trim_t])
                frame_counter = frame_counter_snap + trim_t

                # his_cond / his_eef / registry 전체 복원
                his_cond = list(his_cond_snap)
                his_eef  = list(his_eef_snap)
                registry.restore(registry_snap)

                # SAM3: last_good 위치로 anchor 복원 (복원된 registry 기준)
                for label in object_labels:
                    obj = registry.get(label)
                    if obj.last_good_frame is not None and obj.last_good_mask is not None:
                        ys, xs = np.where(obj.last_good_mask)
                        H, W = obj.last_good_frame.shape[:2]
                        if len(ys) > 0:
                            box_r = [float(xs.min()/W), float(ys.min()/H),
                                     float(xs.max()/W), float(ys.max()/H)]
                            sam_manager.set_anchor(label, obj.last_good_frame, box_r)
                sam_manager.reset_session()

                step_retry_count += 1
                rollback_count   += 1
                continue   # 같은 i 재시도 (i 증가 없음)

            # ── 진행: 성공 or 최대 재시도 초과 ─────────────────────
            step_retry_count = 0

            if neg_event_detected:
                # 최대 재시도 초과 → 안전 프레임만 저장하고 계속
                trim_t = max(0, (first_bad_t or 0) - TRIM_MARGIN)
                n_save = trim_t
                print(f"[MAX_RETRIES] step={i}: saving {n_save} safe frames and continuing")
            elif i == interact_num - 1:
                n_save = len(step_full)
            else:
                n_save = min(pred_step - 1, len(step_full))

            video_to_save.extend(step_full[:n_save])
            vis_frames_to_save.extend(step_vis[:n_save])
            for label in object_labels:
                tracking_log[label].extend(step_log[label][:n_save])
            frame_counter = frame_counter_snap + n_save

            # history buffer 업데이트
            his_eef.append(cartesian_pose[pred_step - 1:pred_step])
            his_cond.append(torch.cat([pred_latents[v][pred_step - 1] for v in range(3)], dim=1).unsqueeze(0))

            i += 1   # 성공 시에만 증가

            # 새 step으로 넘어갈 때 에러 점수 완전 리셋
            for label in object_labels:
                bad_streak[label]  = 0
                error_score[label] = 0.0
                soft_streak[label] = 0

        # ── 저장 ─────────────────────────────────────────────

        # 생성 영상
        video_arr = np.stack(video_to_save, axis=0)
        save_path = f"{base_dir}/online_{uuid}_{val_id_i}_{start_idx_i}_{args.keyboard}.mp4"
        mediapy.write_video(save_path, video_arr, fps=4)
        print(f"\nSaved: {save_path}  (rollbacks={rollback_count}, false_gens={false_gen_count})")

        # SAM3 mask overlay 영상 (tracking 확인용)
        if vis_frames_to_save:
            vis_arr  = np.stack(vis_frames_to_save, axis=0)
            vis_path = f"{base_dir}/online_{uuid}_{val_id_i}_{start_idx_i}_{args.keyboard}_sam3.mp4"
            mediapy.write_video(vis_path, vis_arr, fps=4)
            print(f"SAM3 overlay: {vis_path}")

        # 객체별 tracking log JSON 저장
        for label, log in tracking_log.items():
            safe_label = label.replace(" ", "_").replace("/", "-")
            json_path  = f"{base_dir}/online_{uuid}_{val_id_i}_{start_idx_i}_{args.keyboard}_{safe_label}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    "label":        label,
                    "initial_area": sam_manager.initial_areas.get(label, 0.0),
                    "frames":       log,
                }, f, indent=2)
            print(f"Tracking log: {json_path}")
