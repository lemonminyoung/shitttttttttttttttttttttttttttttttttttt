"""
DROID TF dataset 탐색 스크립트.
각 episode의 첫 step 이미지 3장을 droid_data/result/<ep>_<task>/ 에 저장한다.

cd /home/dgu/minyoung/Ctrl-World
python run_tf_dataset.py
"""

import os
import re
import tensorflow_datasets as tfds
from PIL import Image

DATASET_DIR = "/home/dgu/minyoung/droid_data/droid_100/1.0.0"
RESULT_DIR  = "/home/dgu/minyoung/droid_data/result"

os.makedirs(RESULT_DIR, exist_ok=True)

builder = tfds.builder_from_directory(DATASET_DIR)
ds = builder.as_dataset(split="train")


def safe_str(b) -> str:
    s = b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)
    s = s.strip()
    s = re.sub(r'[^\w\-]', '_', s)
    return s[:60]


for ep_idx, episode in enumerate(tfds.as_numpy(ds)):
    steps = list(episode["steps"])
    step0 = steps[0]
    obs   = step0["observation"]

    lang   = safe_str(step0["language_instruction"])
    ep_dir = os.path.join(RESULT_DIR, f"ep{ep_idx:04d}_{lang}")
    os.makedirs(ep_dir, exist_ok=True)

    Image.fromarray(obs["exterior_image_1_left"]).save(
        os.path.join(ep_dir, "step000_exterior1.png"))
    Image.fromarray(obs["exterior_image_2_left"]).save(
        os.path.join(ep_dir, "step000_exterior2.png"))
    Image.fromarray(obs["wrist_image_left"]).save(
        os.path.join(ep_dir, "step000_wrist.png"))

    print(
        f"[ep{ep_idx:04d}] steps={len(steps):3d} | "
        f"lang='{step0['language_instruction'].decode()}' | "
        f"saved → {ep_dir}"
    )

print(f"\n완료: {ep_idx + 1} episodes → {RESULT_DIR}")
