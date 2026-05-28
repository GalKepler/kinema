"""Center-of-mass trajectory computed from MediaPipe keypoints.

I/O (write_com / read_com / COM_SCHEMA) lives in kinema.io.com — import from there.
"""

# Segment mass fractions from Winter DA (2009) "Biomechanics and Motor Control
# of Human Movement", 4th ed., Table 4.1 (whole-body = 1.0).
#
# Simplifications vs. source table:
#   - Left/right limbs treated symmetrically (same fraction each side).
#   - Hand sub-segments aggregated to wrist→index midpoint.
#   - Foot sub-segments aggregated to ankle→foot_index midpoint.
#   - Head: ear-to-ear midpoint used as proxy (fraction 0.081).
#   - Trunk COM = midpoint of shoulder-midpoint and hip-midpoint;
#     fraction is residual (1.0 - sum of all other segments ≈ 0.497).

import logging
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from kinema.io.com import read_com, write_com  # re-export for pipeline
from kinema.io.keypoints import LANDMARK_NAMES
from kinema.kinematics.smoothing import savgol_smooth

NDArrayF32 = npt.NDArray[np.float32]
NDArrayF64 = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)

# Each tuple: (proximal_landmark, distal_landmark, mass_fraction)
# Bilateral segments appear once per side with the per-side fraction.
_SEGMENT_DEFS: Final[list[tuple[str, str, float]]] = [
    # head (ear-to-ear midpoint)
    ("LEFT_EAR",         "RIGHT_EAR",          0.081),
    # upper arm, per side
    ("LEFT_SHOULDER",    "LEFT_ELBOW",          0.028),
    ("RIGHT_SHOULDER",   "RIGHT_ELBOW",         0.028),
    # forearm, per side
    ("LEFT_ELBOW",       "LEFT_WRIST",          0.016),
    ("RIGHT_ELBOW",      "RIGHT_WRIST",         0.016),
    # hand, per side
    ("LEFT_WRIST",       "LEFT_INDEX",          0.006),
    ("RIGHT_WRIST",      "RIGHT_INDEX",         0.006),
    # thigh, per side
    ("LEFT_HIP",         "LEFT_KNEE",           0.100),
    ("RIGHT_HIP",        "RIGHT_KNEE",          0.100),
    # shank, per side
    ("LEFT_KNEE",        "LEFT_ANKLE",          0.0465),
    ("RIGHT_KNEE",       "RIGHT_ANKLE",         0.0465),
    # foot, per side
    ("LEFT_ANKLE",       "LEFT_FOOT_INDEX",     0.0145),
    ("RIGHT_ANKLE",      "RIGHT_FOOT_INDEX",    0.0145),
]

_EXPLICIT_FRACTION: Final[float] = sum(f for _, _, f in _SEGMENT_DEFS)
_TRUNK_FRACTION: Final[float] = round(1.0 - _EXPLICIT_FRACTION, 6)

# Informational dict — not used by compute_com internally.
SEGMENT_MASS_FRACTIONS: Final[dict[str, float]] = {
    "head":      0.081,
    "trunk":     _TRUNK_FRACTION,
    "upper_arm": 0.028,
    "forearm":   0.016,
    "hand":      0.006,
    "thigh":     0.100,
    "shank":     0.0465,
    "foot":      0.0145,
}

_LM_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(LANDMARK_NAMES)}


def _pivot_wide(
    keypoints: pd.DataFrame,
) -> tuple[NDArrayF32, NDArrayF32, NDArrayF32, NDArrayF32]:
    """Pivot long-form keypoints to (N_frames x N_landmarks) arrays.

    Parameters
    ----------
    keypoints : pd.DataFrame
        Long-form DataFrame conforming to KEYPOINTS_SCHEMA.

    Returns
    -------
    tuple[NDArrayF32, NDArrayF32, NDArrayF32, NDArrayF32]
        Arrays x, y, z, visibility of shape (N_frames, N_landmarks).
        Missing entries are NaN.
    """
    n_lm = len(LANDMARK_NAMES)
    frames = sorted(keypoints["frame_idx"].unique())
    n_frames = len(frames)
    frame_to_row = {f: i for i, f in enumerate(frames)}

    x = np.full((n_frames, n_lm), np.nan, dtype=np.float32)
    y = np.full((n_frames, n_lm), np.nan, dtype=np.float32)
    z = np.full((n_frames, n_lm), np.nan, dtype=np.float32)
    vis = np.full((n_frames, n_lm), np.nan, dtype=np.float32)

    for row in keypoints.itertuples(index=False):
        row_typed: Any = row
        ri = frame_to_row[int(row_typed.frame_idx)]
        ci = int(row_typed.landmark_id)
        x[ri, ci] = float(row_typed.x)
        y[ri, ci] = float(row_typed.y)
        z[ri, ci] = float(row_typed.z)
        vis[ri, ci] = float(row_typed.visibility)

    return x, y, z, vis


def compute_com(keypoints: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Compute smoothed center-of-mass trajectory from MediaPipe keypoints.

    Mass fractions follow Winter (2009) Table 4.1 with bilateral symmetry
    and sub-segment aggregations (see module docstring).

    Parameters
    ----------
    keypoints : pd.DataFrame
        Long-form DataFrame conforming to ``kinema.io.keypoints.KEYPOINTS_SCHEMA``.
    fps : float
        Frames per second; used by Savitzky-Golay smoothing.

    Returns
    -------
    pd.DataFrame
        Columns: frame_idx (int32), timestamp_ms (int64),
        com_x, com_y, com_z (float32, smoothed normalized coordinates).
    """
    x_w, y_w, z_w, _ = _pivot_wide(keypoints)
    frames = np.array(sorted(keypoints["frame_idx"].unique()), dtype=np.int32)
    n_frames = len(frames)

    com_x = np.zeros(n_frames, dtype=np.float64)
    com_y = np.zeros(n_frames, dtype=np.float64)
    com_z = np.zeros(n_frames, dtype=np.float64)
    weight_sum = np.zeros(n_frames, dtype=np.float64)

    def _add_segment(lm_a: str, lm_b: str, frac: float) -> None:
        ia, ib = _LM_INDEX[lm_a], _LM_INDEX[lm_b]
        mx = (x_w[:, ia] + x_w[:, ib]) / 2.0
        my = (y_w[:, ia] + y_w[:, ib]) / 2.0
        mz = (z_w[:, ia] + z_w[:, ib]) / 2.0
        valid = ~(np.isnan(mx) | np.isnan(my) | np.isnan(mz))
        com_x[valid] += frac * mx[valid]
        com_y[valid] += frac * my[valid]
        com_z[valid] += frac * mz[valid]
        weight_sum[valid] += frac

    for prox, dist, frac in _SEGMENT_DEFS:
        _add_segment(prox, dist, frac)

    # trunk: midpoint of shoulder-midpoint and hip-midpoint
    ls, rs = _LM_INDEX["LEFT_SHOULDER"], _LM_INDEX["RIGHT_SHOULDER"]
    lh, rh = _LM_INDEX["LEFT_HIP"], _LM_INDEX["RIGHT_HIP"]
    trunk_mx = (x_w[:, ls] + x_w[:, rs] + x_w[:, lh] + x_w[:, rh]) / 4.0
    trunk_my = (y_w[:, ls] + y_w[:, rs] + y_w[:, lh] + y_w[:, rh]) / 4.0
    trunk_mz = (z_w[:, ls] + z_w[:, rs] + z_w[:, lh] + z_w[:, rh]) / 4.0
    trunk_valid = ~(np.isnan(trunk_mx) | np.isnan(trunk_my) | np.isnan(trunk_mz))
    com_x[trunk_valid] += _TRUNK_FRACTION * trunk_mx[trunk_valid]
    com_y[trunk_valid] += _TRUNK_FRACTION * trunk_my[trunk_valid]
    com_z[trunk_valid] += _TRUNK_FRACTION * trunk_mz[trunk_valid]
    weight_sum[trunk_valid] += _TRUNK_FRACTION

    # renormalize by actual weight accumulated (handles missing landmarks gracefully)
    nonzero = weight_sum > 0
    com_x[nonzero] /= weight_sum[nonzero]
    com_y[nonzero] /= weight_sum[nonzero]
    com_z[nonzero] /= weight_sum[nonzero]
    com_x[~nonzero] = np.nan
    com_y[~nonzero] = np.nan
    com_z[~nonzero] = np.nan

    com_x = savgol_smooth(com_x, fps)
    com_y = savgol_smooth(com_y, fps)
    com_z = savgol_smooth(com_z, fps)

    ts_map = (
        keypoints.groupby("frame_idx")["timestamp_ms"]
        .first()
        .reindex(frames.tolist())
    )
    timestamps = ts_map.to_numpy(dtype=np.int64)

    return pd.DataFrame(
        {
            "frame_idx": frames,
            "timestamp_ms": timestamps,
            "com_x": com_x.astype(np.float32),
            "com_y": com_y.astype(np.float32),
            "com_z": com_z.astype(np.float32),
        }
    )


__all__ = [
    "SEGMENT_MASS_FRACTIONS",
    "compute_com",
    "read_com",
    "write_com",
]
