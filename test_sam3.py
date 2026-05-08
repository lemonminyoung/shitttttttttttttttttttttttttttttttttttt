import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import tempfile
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from sam3.model_builder import build_sam3_video_predictor
from config import wm_args


def get_frames(args, traj_id, start_idx=0, steps=8):
    """get_traj_info()에서 모델 불필요한 VAE 인코딩 부분을 제외하고,
    raw 픽셀 프레임과 annotation 정보만 반환."""
    val_dataset_dir = args.val_dataset_dir
    val_dataset_dir = f"/home/dgu/minyoung/Ctrl-World/dataset_example/droid_subset"  # 실제 경로로 수정
    skip = args.skip_step

    annotation_path = f"{val_dataset_dir}/annotation/val/{traj_id}.json"
    #annotation_path = f"{val_dataset_dir}/annotation/val/{'0004'}.json"
    with open(annotation_path) as f:
        anno = json.load(f)
    try:
        length = len(anno['action'])
    except:
        length = anno["video_length"]

    frames_ids = np.arange(start_idx, start_idx + steps * skip, skip)
    max_ids = np.ones_like(frames_ids) * (length - 1)
    frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
    print("frames_ids:", frames_ids)

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
        video_dict.append(frames[frames_ids])  # (T, H, W, 3) uint8

    return car_action, joint_pos, video_dict, instruction


# --- 설정 ---
TRAJ_ID = '1799'
START_IDX = 23
#TRAJ_ID = '18599'
#START_IDX = 1
VIEW_IDX = 2  # 확인할 카메라 뷰

args = wm_args(task_type='pickplace')

# 실제 파이프라인과 동일한 경로로 첫 프레임 추출 (모델 로드 없음)
eef_gt, joint_pos_gt, video_dict, instruction = get_frames(args, TRAJ_ID, start_idx=START_IDX, steps=8)

print("instruction:", instruction)
print("video_dict[0].shape:", video_dict[0].shape)  # (T, H, W, 3)

# 첫 프레임 저장 (확인용)
first_frame = video_dict[VIEW_IDX][0]  # (H, W, 3) uint8
Image.fromarray(first_frame).save('first_frame_v1.jpg')
print("first_frame shape:", first_frame.shape)

# --- SAM3 ---
vp = build_sam3_video_predictor(
    checkpoint_path='/home/dgu/minyoung/sam3/checkpoints/sam3.pt'
)

tmpdir = tempfile.mkdtemp()
for i in range(min(5, len(video_dict[VIEW_IDX]))):
    Image.fromarray(video_dict[VIEW_IDX][i]).save(f'{tmpdir}/{i:05d}.jpg')

session_id = vp.start_session(tmpdir)['session_id']

prompts = ['robot arm and end-effector' , 'orange pen']
for i, prompt in enumerate(prompts):
    r = vp.add_prompt(session_id, frame_idx=0, text=prompt, obj_id=i + 1)
    masks = r['outputs']['out_binary_masks']
    area = int(masks.sum()) if len(masks) > 0 else 0
    print(f"'{prompt}': found={len(masks) > 0}, area={area}")

    if len(masks) > 0:
        vis = Image.fromarray(first_frame).convert('RGBA')
        overlay = np.zeros((*first_frame.shape[:2], 4), dtype=np.uint8)
        overlay[masks[0]] = [0, 255, 0, 128]
        vis = Image.alpha_composite(vis, Image.fromarray(overlay))
        vis.convert('RGB').save(f'mask_{prompt.replace(" ", "_")}.jpg')
        print(f"  → mask 저장됨")
