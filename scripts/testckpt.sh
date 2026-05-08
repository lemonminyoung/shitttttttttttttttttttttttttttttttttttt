#!/usr/bin/env bash
set -euo pipefail

# 저장 파일명 예:
# scripts/run_phase1_eval_grid.sh
#
# 실행:
# bash scripts/run_phase1_eval_grid.sh

export CUDA_VISIBLE_DEVICES=0,1

CKPTS=(300 600)
SEEDS=(42 10 80)

BASE_OUT="eval_results/phase1_grid_300_600_seeds"
LOG_DIR="${BASE_OUT}/logs"
mkdir -p "${LOG_DIR}"

for CKPT in "${CKPTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "[RUN] checkpoint=${CKPT} seed=${SEED}"
    echo "============================================================"

    OUT_DIR="${BASE_OUT}/ckpt_${CKPT}_seed_${SEED}"
    ADAPTER_CKPT="model_ckpt/phase1_20260502T122400/checkpoint-${CKPT}.pt"
    LOG_FILE="${LOG_DIR}/ckpt_${CKPT}_seed_${SEED}.log"

    python scripts/rollout_interact_pi_online.py \
      --task_type pickplace \
      --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt \
      --svd_model_path checkpoints/svd \
      --clip_model_path checkpoints/clip \
      --pi_ckpt /home/dgu/minyoung/checkpoints/pi05_droid_pytorch \
      --sam3_ckpt /home/dgu/minyoung/sam3/checkpoints/sam3.pt \
      --object_labels "robot arm and end-effector,pen" \
      --view_idx 1 \
      --seed "${SEED}" \
      --phase1_eval_mode \
      --phase1_eval_steps 1 \
      --phase1_eval_out "${OUT_DIR}" \
      --use_obj_token \
      --adapter_ckpt "${ADAPTER_CKPT}" \
      2>&1 | tee "${LOG_FILE}"

    echo "[DONE] checkpoint=${CKPT} seed=${SEED}"
    echo "output: ${OUT_DIR}"
    echo "log:    ${LOG_FILE}"
  done
done

echo "============================================================"
echo "[ALL DONE] Results saved under ${BASE_OUT}"
echo "============================================================"

python - <<'PY'
import json
from pathlib import Path
from statistics import mean, pstdev

base = Path("eval_results/phase1_grid_300_600_seeds")
metrics_files = sorted(base.glob("ckpt_*_seed_*/traj_*_start_*/step_*/metrics.json"))

rows = []
for p in metrics_files:
    with open(p) as f:
        m = json.load(f)

    # ckpt/seed는 폴더명에서 추출
    run_dir = p.parts[len(base.parts)]
    # 예: ckpt_300_seed_42
    parts = run_dir.split("_")
    ckpt = int(parts[1])
    seed = int(parts[3])

    row = {
        "ckpt": ckpt,
        "seed": seed,
        "path": str(p),
        "latent_mse_correct_vs_zero": m.get("latent_mse_correct_vs_adapter_zero"),
        "latent_mse_correct_vs_shifted": m.get("latent_mse_correct_vs_adapter_shifted"),
        "latent_mse_zero_vs_baseline": m.get("latent_mse_zero_vs_baseline_no_obj"),
        "pixel_mse_correct_vs_zero": m.get("pixel_mse_correct_vs_adapter_zero"),
        "pixel_mse_correct_vs_shifted": m.get("pixel_mse_correct_vs_adapter_shifted"),
        "obj_bg_ratio_correct_vs_zero": m.get("obj_bg_ratio_correct_vs_adapter_zero"),
        "obj_bg_ratio_correct_vs_shifted": m.get("obj_bg_ratio_correct_vs_adapter_shifted"),
    }
    rows.append(row)

if not rows:
    print("[WARN] No metrics.json found.")
    raise SystemExit(0)

summary = {}
for ckpt in sorted(set(r["ckpt"] for r in rows)):
    rs = [r for r in rows if r["ckpt"] == ckpt]
    summary[str(ckpt)] = {"num_runs": len(rs), "metrics": {}}

    keys = [
        "latent_mse_correct_vs_zero",
        "latent_mse_correct_vs_shifted",
        "latent_mse_zero_vs_baseline",
        "pixel_mse_correct_vs_zero",
        "pixel_mse_correct_vs_shifted",
        "obj_bg_ratio_correct_vs_zero",
        "obj_bg_ratio_correct_vs_shifted",
    ]

    for k in keys:
        vals = [r[k] for r in rs if r[k] is not None]
        if vals:
            summary[str(ckpt)]["metrics"][k] = {
                "mean": mean(vals),
                "std": pstdev(vals) if len(vals) > 1 else 0.0,
                "values": vals,
            }

out_path = base / "summary_300_600_seeds.json"
with open(out_path, "w") as f:
    json.dump({"runs": rows, "summary": summary}, f, indent=2)

print(f"[SUMMARY] wrote {out_path}")
print(json.dumps(summary, indent=2))
PY
