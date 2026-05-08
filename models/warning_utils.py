"""
Warning score computation for Ctrl-World object-state conditioning.

Hard warning : crushed / vanished  → retrospective prefix (hard_horizon=2)
Soft warning : occluded            → forward persistence  (soft_persistence=4)

warning_vec (8-dim):
  [hard_any, soft_any, hard_count, soft_count,
   hard_max, soft_max, hard_sum,  soft_sum]
"""

import numpy as np

HARD_HORIZON     = 2
SOFT_PERSISTENCE = 4
HARD_THRESHOLD   = 2.0
SOFT_THRESHOLD   = 1.5
WARNING_DIM      = 8


def compute_object_scores(record: dict) -> tuple:
    """
    H_i, S_i, hard_i, soft_i for one object × one frame.

    record keys: cause, error_score, bad_streak, shape_rejected,
                 area_ratio, extent_ratio, state, iou, absent
    """
    cause         = record.get('cause') or ''
    error_score   = float(record.get('error_score') or 0.0)
    bad_streak    = int(record.get('bad_streak') or 0)
    shape_rejected = bool(record.get('shape_rejected', False))
    area_ratio    = float(record.get('area_ratio', 1.0))
    extent_ratio  = float(record.get('extent_ratio', 1.0))
    state         = float(record.get('state', 0.0))
    iou           = record.get('iou')
    absent        = bool(record.get('absent', False))

    H_i = (
        2.0  * float(cause in ('crushed', 'vanished')) +
        1.0  * min(error_score, 2.0) +
        0.5  * min(bad_streak, 2) +
        0.75 * float(shape_rejected) +
        0.5  * float(area_ratio  < 0.70) +
        0.5  * float(extent_ratio < 0.70)
    )
    hard_i = int(H_i >= HARD_THRESHOLD)

    S_i = (
        1.5  * float(cause == 'occluded') +
        0.5  * float(state == 1.0) +
        0.5  * float(iou is not None and float(iou) > 0.05) +
        0.25 * float(absent) -
        1.0  * float(hard_i == 1)
    )
    soft_i = int(S_i >= SOFT_THRESHOLD)
    if hard_i == 1:
        soft_i = 0

    return float(H_i), float(S_i), hard_i, soft_i


def compute_warning_vec(object_records: list) -> np.ndarray:
    """
    list[dict] (one dict per object) → (8,) float32 warning vector.
    Permutation-invariant: max / sum aggregation over objects.
    """
    if not object_records:
        return np.zeros(WARNING_DIM, dtype=np.float32)

    results    = [compute_object_scores(r) for r in object_records]
    H_vals     = [r[0] for r in results]
    S_vals     = [r[1] for r in results]
    hard_labels = [r[2] for r in results]
    soft_labels = [r[3] for r in results]

    return np.array([
        float(max(hard_labels)),    # hard_any
        float(max(soft_labels)),    # soft_any
        float(sum(hard_labels)),    # hard_count
        float(sum(soft_labels)),    # soft_count
        float(max(H_vals)),         # hard_max
        float(max(S_vals)),         # soft_max
        float(sum(H_vals)),         # hard_sum
        float(sum(S_vals)),         # soft_sum
    ], dtype=np.float32)


def apply_hard_horizon(vecs: list, hard_horizon: int = HARD_HORIZON) -> list:
    """
    Hard event at t → propagate hard fields backward to [t-hard_horizon, t-1].
    """
    result = [v.copy() for v in vecs]
    for t, v in enumerate(vecs):
        if v[0] > 0:  # hard_any
            for dt in range(1, hard_horizon + 1):
                s = t - dt
                if s >= 0:
                    result[s][0] = max(result[s][0], 1.0)   # hard_any
                    result[s][2] = max(result[s][2], v[2])   # hard_count
                    result[s][4] = max(result[s][4], v[4])   # hard_max
                    result[s][6] = max(result[s][6], v[6])   # hard_sum
    return result


def apply_soft_persistence(vecs: list, soft_persistence: int = SOFT_PERSISTENCE) -> list:
    """
    Soft event at t → maintain soft fields forward to [t+1, t+soft_persistence].
    """
    result = [v.copy() for v in vecs]
    T = len(vecs)
    for t, v in enumerate(vecs):
        if v[1] > 0:  # soft_any
            for dt in range(1, soft_persistence + 1):
                s = t + dt
                if s < T:
                    result[s][1] = max(result[s][1], 1.0)   # soft_any
                    result[s][3] = max(result[s][3], v[3])   # soft_count
                    result[s][5] = max(result[s][5], v[5])   # soft_max
                    result[s][7] = max(result[s][7], v[7])   # soft_sum
    return result


def process_episode_warnings(
    frame_records: list,
    hard_horizon: int = HARD_HORIZON,
    soft_persistence: int = SOFT_PERSISTENCE,
) -> list:
    """
    Full pipeline for one episode.

    frame_records : list[list[dict]]  — outer: frames, inner: objects per frame
    returns       : list of (8,) float32 arrays, one per frame
    """
    raw  = [compute_warning_vec(objs) for objs in frame_records]
    vecs = apply_hard_horizon(raw, hard_horizon)
    vecs = apply_soft_persistence(vecs, soft_persistence)
    return vecs
