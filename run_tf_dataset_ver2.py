"""
선택된 에피소드의 exterior_image_2_left 전체 프레임을 PNG 시퀀스로 저장.
저장 위치: droid_data/result_select_ver2/ep{idx}_{task}/frame_{n:04d}.png

cd /home/dgu/minyoung/Ctrl-World
python run_tf_dataset_ver2.py
"""

import os
import re
import tensorflow_datasets as tfds
from PIL import Image
import imageio

DATASET_DIR = "/home/dgu/minyoung/droid_data/droid_100/1.0.0"
RESULT_DIR  = "/home/dgu/minyoung/droid_data/result_select_ver2"

TARGET_EPS = {0, 1, 3, 4, 5, 21, 36, 54, 60, 74, 80, 86, 91, 92}

os.makedirs(RESULT_DIR, exist_ok=True)

builder = tfds.builder_from_directory(DATASET_DIR)
ds = builder.as_dataset(split="train")


def safe_str(b) -> str:
    s = b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)
    s = s.strip()
    s = re.sub(r'[^\w\-]', '_', s)
    return s[:60]


for ep_idx, episode in enumerate(tfds.as_numpy(ds)):
    if ep_idx not in TARGET_EPS:
        continue

    steps = list(episode["steps"])
    lang  = safe_str(steps[0]["language_instruction"])

    ep_dir = os.path.join(RESULT_DIR, f"ep{ep_idx:04d}_{lang}")
    os.makedirs(ep_dir, exist_ok=True)

    frames = []
    for frame_idx, step in enumerate(steps):
        img = step["observation"]["exterior_image_2_left"]
        Image.fromarray(img).save(
            os.path.join(ep_dir, f"frame_{frame_idx:04d}.png"))
        frames.append(img)

    mp4_path = os.path.join(ep_dir, "video.mp4")
    imageio.mimwrite(mp4_path, frames, fps=10)

    print(f"[ep{ep_idx:04d}] {len(steps)} frames + video.mp4 → {ep_dir}")

    if ep_idx >= max(TARGET_EPS):
        break  # 마지막 타겟 에피소드(92) 이후 종료

print(f"\n완료 → {RESULT_DIR}")
