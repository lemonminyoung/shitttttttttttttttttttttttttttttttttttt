import torch
import os
import json
from dataclasses import dataclass


@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths
    svd_model_path = "/home/dgu/minyoung/Ctrl-World/checkpoints/svd"
    clip_model_path = "/home/dgu/minyoung/Ctrl-World/checkpoints/clip"
    ckpt_path = '/home/dgu/minyoung/Ctrl-World/checkpoints/ctrl-world/checkpoint-10000.pt'
    pi_ckpt = '/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid'

    # dataset parameters
    # raw data
    dataset_root_path = "dataset_example"
    dataset_names = 'droid_subset'
    # meta info
    dataset_meta_info_path = 'dataset_meta_info' #'/cephfs/cjyyj/code/video_evaluation/exp_cfg'#'dataset_meta_info'
    dataset_cfgs = dataset_names
    prob=[1.0]
    annotation_name='annotation' #'annotation_all_skip1'
    num_workers=4
    down_sample=1 # downsample 15hz to 5hz
    skip_step = 1


    # logs parameters
    debug = False
    tag = 'droid_subset'
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "droid_example"


    # training parameters
    learning_rate= 1e-5 # 5e-6
    gradient_accumulation_steps = 1
    mixed_precision = 'fp16'
    train_batch_size = 4
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 500000
    checkpointing_steps = 20000
    validation_steps = 2500
    max_grad_norm = 1.0
    # for val
    video_num= 10

    ############################ model args ##############################

    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 1.0 #2.0 #7.5 #7.5 #7.5 #3.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    # num history and num future predictions
    num_frames= 5
    num_history = 6
    action_dim = 7
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.bfloat16 # [torch.float32, torch.bfloat16] # during inference, we can use bfloat16 to accelerate the inference speed and save memory
    use_object_state = False   # object-state adapter 활성화 (shape_projector + ObjectStateEncoder)
    obj_dropout_prob = 0.5     # ObjectStateEncoder 학습 시 dropout 확률
    use_warning      = False   # warning conditioning 활성화 (WarningEncoder)
    use_hard_warning = True    # hard warning (crushed/vanished) 사용
    use_soft_warning = True    # soft warning (occluded) 사용



    ########################### rollout args ############################
    # policy
    task_type: str = "pickplace" # choose from ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','tissue','drawer','stack']
    gripper_max_dict = {'replay':1.0, 'pickplace':0.75, 'towel_fold':0.95, 'wipe_table':0.95, 'tissue':0.97, 'close_laptop':0.95,'drawer':0.6,'stack':0.75,}
    z_min_dict = {'pickplace':0.23}
    ##############################################################################
    policy_type = 'pi05' # choose from ['pi05', 'pi0', 'pi0fast']
    action_adapter = 'models/action_adapter/model2_15_9.pth' # adapat action from joint vel to cartesian pose
    pred_step = 5 # predict 5 steps (1s) action each time
    policy_skip_step = 2 # horizon = (pred_step-1) * policy_skip_step
    interact_num = 12 # number of interactions (each interaction contains pred_step steps)

    # wm
    data_stat_path = 'dataset_meta_info/droid/stat.json'
    val_model_path = ckpt_path
    history_idx = [0,0,-12,-9,-6,-3]

    # save
    save_dir = 'synthetic_traj'

    # select different traj for different tasks
    def __post_init__(self):
        # Per-task gripper max
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        self.z_min = self.z_min_dict.get(self.task_type, 0.18)
        # Default task_name
        self.task_name = f"Rollouts_interact_pi"
        if self.task_type == "replay":
            self.task_name = "Rollouts_replay"

        # Configure per-task eval sets
        #if self.task_type == "replay":
        #    self.val_dataset_dir = "dataset_example/droid_subset"
        #    self.val_id = ["899", "18599","199","1799"]
        #    self.start_idx = [8, 14, 8] * len(self.val_id)
        #    self.instruction = [""] * len(self.val_id)
        #    self.task_name = "Rollouts_replay"

        if self.task_type == "replay":
            self.val_dataset_dir = "dataset_example/rh20t"
            self.val_id = ["0004"]
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["Pick up the small block on the left and move it to the right"] * len(self.val_id)
            self.task_name = "Rollouts_replay"

        elif self.task_type == "keyboard":
            self.val_dataset_dir = "dataset_example/droid_subset"
            self.val_id = ["18599"]
            self.start_idx = [14] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_keyboard"

        # elif self.task_type == "keyboard2":
        #     self.val_dataset_dir = "/cephfs/shared/droid_hf/droid_svd_v2"
        #     self.val_id = ["1499"]*100
        #     self.start_idx = [8] * len(self.val_id) # 2599 8 #9499 10
        #     self.instruction = [""] * len(self.val_id)
        #     self.task_name = "Rollouts_keyboard_1499"
        #     self.ineraction_num = 7

        elif self.task_type == "pickplace":
            #self.interact_num = 20
            # self.val_dataset_dir = "dataset_example/droid_new_setup"
            # self.val_id = ['0001','0002','0003']
            # self.start_idx = [0] * len(self.val_id)
            # self.instruction = [
            #     "pick up the green block and place in plate",
            #     "pick up the green block and place in plate",
            #     "pick up the blue block and place in plate",]

            '''self.val_dataset_dir = '/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0914/droid_pi05'
            self.val_id = [203038,203715,203803,203837,204021,204112,204202,204331,204437,204502]
            self.start_idx = [0]*len(self.val_id)
            self.instruction = ['pick up the blue block and place in white plate', 'pick up the blue block and place in white plate', 'pick up the blue block and place in white plate',
                                'pick up the blue block and place in white plate', 'pick up the blue block and place in white plate', 'pick up the green block and place in white plate',
                                'pick up the green block and place in white plate', 'pick up the green block and place in white plate', 'pick up the red block and place in white plate',
                                'pick up the red block and place in white plate']'''

            self.val_dataset_dir = 'dataset_example/droid_subset'
            #self.val_dataset_dir = 'dataset_example/droid_new_setup_full/drawer'
            #self.val_id = ['1799']
            #self.val_id = ['0000']
            self.val_id = ['18599']
            self.start_idx = [24]*len(self.val_id) #23 , 0, 24
            self.instruction = ['Pick the black thing from the chair and place it on the table']
            #self.instruction = ['pick the blue block and place it in plate']
            #self.instruction = ['pick up the pen']
            #self.instruction = ['pick the sponge and place it in drawer']

        elif self.task_type == "towel_fold":
            self.interact_num = 15
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id =['0004','0005']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["fold the towel"] * len(self.val_id)

            self.val_dataset_dir = 'dataset_example/droid_new_setup_eval/towel_fold'
            self.val_id = ['000018', '000044', '000120', '000228', '000255', '000336', '000403', '000427', '000453', '000643', '000739', '000803', '000833', '000902', '235555', '235713', '235826', '235933']
            self.start_idx = [0]*len(self.val_id)
            self.instruction = ['fold the towel']*len(self.val_id)

        elif self.task_type == "wipe_table":
            # self.val_dataset_dir = "dataset_example/droid_new_setup"
            # self.val_id = ['0006','0007']
            # self.start_idx = [0] * len(self.val_id)
            # self.instruction = [
            #     "move the towel from left to right",
            #     "move the towel from left to right"
            # ]
            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0918/droid_pi05"
            self.val_id = ['134750', '134908', '135009', '135048', '135205']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ['moving the towel from left to right', 'moving the towel from right to left', 'moving the towel from left to right','moving the towel from left to right','moving the towel from left to right']

        elif self.task_type == "tissue":
            # self.interact_num = 10
            # self.val_dataset_dir = "dataset_example/droid_new_setup"
            # self.val_id = ['0008','0009']
            # self.start_idx = [0] * len(self.val_id)
            # self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            # self.policy_skip_step = 3

            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0918/droid_pi05"
            self.val_id = ['135334', '135425', '135525', '135623']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ['pull one tissue out of the box']*len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0918/droid_pi05"
            self.val_id = ['135334', '135425', '135525', '135623']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ['pull one tissue out of the box']*len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = "/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0922/droid_pi05"
            self.val_id = ['213026','213128','213222','213333','213535']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ['pull one tissue out of the box']*len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "close_laptop":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ['0010','0011']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = "dataset_example/droid_new_setup_eval/laptop"
            self.val_id = ['135749','135849','135931','175856','175930','180035']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "stack":
            self.val_dataset_dir = "dataset_example/droid_new_setup"
            self.val_id = ['0012','0013']
            self.start_idx = [5] * len(self.val_id)
            self.instruction = ["stack the blue block on the red block"] * len(self.val_id)

            self.val_dataset_dir = "dataset_example/droid_new_setup_eval/stack"
            self.val_id = ['163907','164016','164350','232817','233512','234632','234823']
            self.start_idx = [10] * len(self.val_id)
            self.instruction = ["stack the blue block on the red block","stack the blue block on the red block","stack the blue block on the red block","stack the blue block on the red block","stack the green block on the red block","stack the blue block on the green block","stack the blue block on the green block"]

        elif self.task_type == 'drawer':
            self.val_dataset_dir = '/cephfs/shared/droid_hf/data_iclr/droid_real_all_iclr/droid_real0913/droid_pi05'
            self.val_id = [224640,224723,224832,225306,234949]
            self.start_idx = [10]*len(self.val_id)
            self.instruction = ['pick up the sponge and place in the drawer', 'pick up the sponge and place in the drawer', 'pick up the sponge and place in the drawer', 'pick up the sponge and place in the drawer', 'pick up the sponge and place in the drawer']
            self.policy_skip_step = 3

        elif self.task_type == "droid_tracking":
            # ── ACTIVE episode: 아래 3줄만 수정. CLI --val_id로도 덮어쓸 수 있음 ──────
            # object_labels 참고: 각 episode 주석 확인
            self.val_id      = ["54"]      # ACTIVE ← 여기만 수정
            self.start_idx   = [24]
            self.instruction = ["Put the yellow block in the blue cup"] # ACTIVE ← 여기만 수정

            # ── 사용 가능한 episodes (비활성) ────────────────────────
            # ep 28   --object_labels "robot arm and end-effector,white bowl"         start=28  instruction="Put the marker inside the white bowl"
            # ep 44   --object_labels "robot arm and end-effector,green cloth"        start=35  instruction="Move the green cloth to the right"
            # ep 70   --object_labels "robot arm and end-effector,bowl,plate"         start=30  instruction="Remove the bowls from the plate"
            # ep 85   --object_labels "robot arm and end-effector,orange can,bowl"    start=0   instruction="Remove the orange can from the plastic bowl and put it on the table"
            # ep 118  --object_labels "robot arm and end-effector,marker,mug"         start=0   instruction="Remove the marker from the mug"
            # ep 123  --object_labels "robot arm and end-effector,yellow marker,red mug" start=0 instruction="Put the yellow marker inside the red mug"
            # ep 865  --object_labels "robot arm and end-effector,pink plate"         start=0   instruction="Put the pink plate on the table"
            # ── droid_20chunks에서 추가한 episodes ──────────────────
            # ep 54   --object_labels "robot arm and end-effector,yellow block,blue cup"  start=0  instruction="Put the yellow block in the blue cup"
            # ep 1021 --object_labels "robot arm and end-effector,blue ring,wooden tray"  start=0  instruction="Pick up the blue ring from the table and put it in the wooden tray"
            # ep 3024 --object_labels "robot arm and end-effector,block,toy plate rack"   start=0  instruction="Put the block in the toy plate rack"
            # ep 3152 --object_labels "robot arm and end-effector,shaving stick"          start=0  instruction="Put the shaving stick on the right side of the sink"
            # ep 3181 --object_labels "robot arm and end-effector,lime green block,pink block" start=0 instruction="Pick up the lime green block and put it on the pink block"
            # ep 11049 --object_labels "robot arm and end-effector,cup"                   start=0  instruction="Unstack the cups"
            # ep 11051 --object_labels "robot arm and end-effector,pink t-shirt,radish"   start=0  instruction="Move the pink t-shirt to the right, move the radish and white rope to the left"
            # ep 11092 --object_labels "robot arm and end-effector,green thing,black bowl" start=0 instruction="Put the green thing in the black bowl"
            # ep 11095 --object_labels "robot arm and end-effector,maize cob"             start=0  instruction="Pick up the maize cob and move it forward to the right"
            # ep 11126 --object_labels "robot arm and end-effector,pen,cup"               start=0  instruction="Put the pen in the cup"

            self.val_dataset_dir = "/home/dgu/minyoung/droid_data/droid_tracking_dataset"

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

'''
# ============================================================
# Object tracking JSON field 설명
# ============================================================

# ------------------------------------------------------------
# Top-level fields
# ------------------------------------------------------------

label
# 추적 중인 object 이름.
# 예: "pen", "marker", "mug", "robot arm and end-effector"

initial_area
# 첫 frame에서 잡힌 object mask의 픽셀 면적.
# 이후 area_ratio 계산의 기준값으로 사용됨.
# 예: initial_area=84면, 현재 area=42일 때 area_ratio=0.5


# ------------------------------------------------------------
# Per-frame metadata
# ------------------------------------------------------------

frame
# 저장된 rollout video 기준 frame 번호.
# 사람이 overlay/video를 볼 때 사용하는 전체 frame index.

local_idx
# 원본 trajectory 또는 현재 rollout chunk 내부에서의 local index.
# frame과 비슷하지만, 데이터 로딩/skip/chunk 기준 index라서 frame과 항상 같지는 않을 수 있음.

traj_id
# 어떤 trajectory에서 나온 결과인지 나타내는 ID.
# 예: "1799"

task_id
# task 식별자.
# 현재는 traj_id와 같은 값으로 저장하는 경우가 많음.

task_type
# task 종류.
# 예: "pickplace", "droid_tracking", "towel_fold"

text
# language instruction.
# 현재 frame rollout에 사용된 자연어 명령.
# 주의: 첨부 JSON에서는 label은 pen인데 text가 "Remove the marker from the mug"로 되어 있음.
# 이건 명령어 overwrite가 잘못됐을 가능성이 있으니 확인 필요.

view_idx
# multi-view 중 몇 번째 view인지.
# Ctrl-World는 여러 camera view를 세로로 stack해서 쓰므로 view_idx가 중요함.

seed
# rollout 난수 seed.
# 생성 결과 재현성을 위해 저장.

start_idx
# 원본 trajectory에서 rollout을 시작한 frame index.


# ------------------------------------------------------------
# Current object mask / bbox fields
# ------------------------------------------------------------

area
# 현재 frame에서 잡힌 object mask의 픽셀 면적.
# object가 작아지거나 커졌는지 보는 가장 기본적인 값.

bbox
# 현재 object mask를 둘러싼 bounding box.
# pixel 좌표계 [x1, y1, x2, y2].
# 예: [172, 114, 183, 127]

bbox_norm
# bbox를 이미지 크기로 나눈 normalized bbox.
# [x1/W, y1/H, x2/W, y2/H].
# ObjectStateEncoder에 넣기 좋은 좌표 형식.

bbox_area
# bbox의 픽셀 면적.
# 계산식: (x2 - x1) * (y2 - y1)
# mask area가 아니라 bbox rectangle 면적임.

bbox_area_ratio
# 현재 bbox_area / initial bbox_area.
# bbox 크기가 초기보다 얼마나 변했는지 보는 값.
# 1.0이면 초기 bbox 크기와 비슷함.
# 0.5면 bbox가 절반 정도로 작아짐.
# 2.0이면 bbox가 두 배 정도 커짐.

bbox_aspect
# bbox의 가로세로비.
# 계산식: width / height.
# 긴 물체가 갑자기 정사각형처럼 변하거나, 반대로 납작해지는지 보는 보조 신호.

bbox_center
# bbox 중심 좌표.
# [(x1+x2)/2, (y1+y2)/2]

bbox_jump_px
# 이전 frame bbox_center와 현재 frame bbox_center 사이의 픽셀 거리.
# tracking이 갑자기 튀었는지 보는 보조 로그.
# 첫 frame은 이전 bbox가 없으므로 null이 정상.
# 단독 failure 판정용 metric은 아님.


# ------------------------------------------------------------
# Shape / appearance fields
# ------------------------------------------------------------

shape_latent
# 현재 mask shape를 고정 크기 grid로 압축한 vector.
# 값이 0/1 형태라면 downsampled binary mask에 가까운 표현.
# ObjectStateEncoder의 object shape 입력으로 사용 가능.

appearance
# 현재 object crop과 기준 object crop 사이의 appearance similarity.
# 1에 가까울수록 비슷하다고 해석.
# 단, 작은 물체에서는 신뢰도가 낮을 수 있음.

shape_score
# 현재 mask shape가 기준 shape와 얼마나 비슷한지 나타내는 점수.
# 1에 가까울수록 기준 shape와 유사.
# 낮으면 현재 mask를 last_good memory로 업데이트하면 위험함.

shape_rejected
# shape_score 또는 shape 관련 조건 때문에 현재 mask를 신뢰하지 않겠다는 flag.
# True이면 현재 mask를 last_good으로 갱신하지 않는 것이 안전함.

area_ratio
# 현재 mask area / initial mask area.
# 1.0이면 초기 mask 면적과 비슷함.
# 0.5면 object mask가 초기의 절반 정도로 줄어든 것.
# 작은 object에서는 몇 픽셀 차이만으로도 많이 흔들릴 수 있음.

extent_ratio
# mask가 bbox 안을 얼마나 채우는지 보는 값의 비율.
# 보통 object mask의 compactness / fill 정도를 보는 보조 지표.
# 값이 너무 낮으면 bbox는 있는데 mask가 빈약하게 잡혔을 가능성이 있음.


# ------------------------------------------------------------
# Action / interaction fields
# ------------------------------------------------------------

action
# 현재 frame 또는 현재 rollout step에 대응되는 robot action.
# 보통 7차원:
# [x, y, z, rot_x, rot_y, rot_z, gripper] 또는 유사한 end-effector action.
# 정확한 의미는 action adapter / policy output 정의를 따라야 함.

iou
# robot arm mask와 object mask의 overlap 정도.
# object와 로봇이 접촉/상호작용 중인지 보는 보조 신호.
# 0이면 overlap이 없음.

state
# object interaction state.
# 보통 0/1 값으로 사용.
# 1이면 robot과 object가 접촉/상호작용 중이라고 해석 가능.
# 0이면 아직 직접 상호작용 중이 아님.


# ------------------------------------------------------------
# Failure / rollback fields
# ------------------------------------------------------------

absent
# object가 현재 frame에서 사라졌다고 판단했는지.
# True이면 object missing / detection lost 가능성.

cause
# failure 원인 문자열.
# 정상 frame이면 null.
# 예: "absent", "shape_score_low", "area_ratio_low" 등으로 확장 가능.

recovery_tier
# recovery/rollback 단계 수준.
# 0이면 별도 recovery 없음.
# 값이 커질수록 강한 recovery를 시도한 것으로 해석 가능.

rollback_candidate
# 이 frame이 rollback 후보인지.
# True이면 현재 생성이 나빠졌다고 보고 이전 good state로 되돌릴 후보.

redetected
# object가 lost된 뒤 다시 detector로 재탐지되었는지.
# True이면 tracking continuity가 끊겼을 가능성이 있으므로 audit에서 확인 필요.

soft_recovered
# 강한 rollback은 아니지만 soft recovery가 적용되었는지.
# 예: 이전 memory를 참고해 완화 보정한 경우.

bad_streak
# 연속으로 bad/risk 상태가 나온 횟수.
# 값이 커질수록 object tracking/generation이 계속 불안정하다는 뜻.

error_score
# 현재 frame의 종합 error 점수.
# 0이면 현재 로직상 큰 문제 없음.
# 단, 기존 metric이 둔감하면 실제 실패가 있어도 0일 수 있음.

update_rejected
# 현재 frame의 object 정보를 last_good memory로 업데이트하지 않았는지.
# True이면 현재 mask/object state를 신뢰하지 않겠다는 뜻.

update_reject_reason
# update_rejected=True가 된 이유.
# 예: "shape_score_low", "area_ratio_low", "absent"
# null이면 update 거부 이유 없음.

phase2_candidate
# Phase2 학습 후보로 사람이 확인해볼 frame인지.
# True라고 해서 바로 학습에 넣으면 안 됨.
# 최종적으로는 audit_label을 사람이 보고 확정해야 함.

candidate_reason
# phase2_candidate=True가 된 이유.
# 예: "shape_score_low", "area_ratio_low", "rollback_candidate"


# ------------------------------------------------------------
# Detector / coordinate domain fields
# ------------------------------------------------------------

detector_domain
# mask를 어떤 domain에서 검출했는지.
# "original": 원본 해상도 frame에서 검출
# "super_resolution": Real-ESRGAN 등으로 키운 frame에서 검출

downstream_domain
# registry/model에 실제로 넘기는 좌표계.
# 보통 "original"이어야 함.
# SR에서 검출했더라도 mask/bbox는 원본 해상도로 내려서 사용해야 함.

scale_back_applied
# SR domain에서 얻은 mask/bbox를 원본 해상도로 되돌렸는지.
# True이면 SR detection 후 original coordinate로 scale-back했다는 뜻.

scale_factor
# SR 배율.
# 예: 4.0이면 4x upsample frame에서 detector를 돌렸다는 뜻.


# ------------------------------------------------------------
# Last-good object memory fields
# ------------------------------------------------------------

last_good_frame
# 현재 frame 이전에 신뢰 가능하다고 판단된 마지막 good frame 번호.
# frame 0은 이전 good frame이 없으므로 null이 정상.

last_good_area
# last_good_frame에서의 object mask area.

last_good_bbox
# last_good_frame에서의 bbox pixel 좌표.
# [x1, y1, x2, y2]

last_good_bbox_norm
# last_good_frame에서의 normalized bbox.
# Phase2에서 last_good object state를 다시 넣을 때 중요함.

last_good_shape_latent
# last_good_frame에서의 shape_latent.
# Phase2에서 “망가지기 전 object shape memory”로 사용할 수 있음.

last_good_state
# last_good_frame에서의 interaction state.
# object가 당시 robot과 접촉 중이었는지 등의 정보.

last_good_appearance
# last_good_frame에서의 appearance similarity.

last_good_updated
# 현재 frame이 audit 기준 good이라서 last_good memory로 업데이트되었는지.
# True이면 현재 frame이 다음 frame들의 last_good 기준이 될 수 있음.
# 주의: 실제 registry 내부 update와 완전히 같은 의미가 아닐 수 있고,
# logging/audit 기준 update flag로 보는 게 안전함.


# ------------------------------------------------------------
# Human audit fields
# ------------------------------------------------------------

tracking_status
# 자동으로 붙인 tracking 상태 요약.
# "tracking_ok": 현재 metric상 tracking이 안정적
# "tracking_suspicious": shape/area/bbox 등이 의심스러워 사람이 확인 필요
# "tracking_lost": object가 없거나 bbox가 없음
# 주의: generation failure 확정 label이 아님.

generation_status
# 생성 품질 상태.
# 현재는 자동 확정하지 않고 "unknown"으로 두는 게 안전함.
# 사람이 overlay를 보고 generation_bad / generation_ok로 바꿀 수 있음.

audit_label
# 사람이 최종으로 붙일 label.
# Phase2 학습에는 이 값이 가장 중요함.
# 예:
# "tracking_ok_generation_ok"
# "tracking_ok_generation_bad"
# "tracking_bad_generation_unknown"
# "tracking_lost"
# "uncertain"
# 처음 저장 시에는 "todo"가 정상.'''
