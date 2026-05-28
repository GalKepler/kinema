"""Move segmentation from COM velocity.

CSV columns written by :func:`write_moves`:

    index           - move number (1-based)
    start_frame     - frame_idx of first frame in move
    end_frame       - frame_idx of last frame in move
    start_time_sec  - timestamp of first frame in seconds
    end_time_sec    - timestamp of last frame in seconds
    duration_sec    - end_time_sec - start_time_sec
    peak_speed      - max COM speed over move (normalized units/sec)
    mean_speed      - mean COM speed over move (normalized units/sec)
    normalized_jerk - dimensionless jerk score (lower = smoother)

Algorithm
---------
1. Differentiate smoothed COM to get per-frame speed (‖velocity‖).
2. Smooth speed with a 0.5 s window to reduce quantization noise.
3. Threshold at ``threshold_quantile`` of the smoothed speed signal.
4. Find maximal above-threshold runs; drop those shorter than
   ``min_move_duration_sec``.
5. Merge consecutive runs separated by fewer than ``min_rest_duration_sec``
   frames; re-apply duration filter after merging.

This heuristic is intentionally crude — appropriate for the MVP.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from kinema.kinematics.metrics import segment_jerk
from kinema.kinematics.smoothing import savgol_smooth

logger = logging.getLogger(__name__)

NDArrayF64 = npt.NDArray[np.float64]


@dataclass
class Move:
    """A discrete movement segment detected in a COM trajectory."""

    index: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    peak_speed: float
    mean_speed: float
    normalized_jerk: float


def _compute_speed(com: pd.DataFrame, fps: float) -> NDArrayF64:
    """Differentiate COM and return per-frame speed magnitude (normalized units/sec).

    Parameters
    ----------
    com : pd.DataFrame
        DataFrame with ``com_x``, ``com_y``, ``com_z`` columns.
    fps : float
        Frames per second.

    Returns
    -------
    NDArrayF64
        1-D speed array, shape (N,).
    """
    dt = 1.0 / fps
    x = com["com_x"].to_numpy(dtype=np.float64)
    y = com["com_y"].to_numpy(dtype=np.float64)
    z = com["com_z"].to_numpy(dtype=np.float64)
    vx: NDArrayF64 = np.gradient(x, dt)
    vy: NDArrayF64 = np.gradient(y, dt)
    vz: NDArrayF64 = np.gradient(z, dt)
    result: NDArrayF64 = np.sqrt(vx**2 + vy**2 + vz**2)
    return result


def _find_runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return (start, end) inclusive positional index pairs for True runs.

    Parameters
    ----------
    mask : NDArrayBool
        1-D boolean array.

    Returns
    -------
    list[tuple[int, int]]
        Each tuple is (start_inclusive, end_inclusive).
    """
    runs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _merge_runs(
    runs: list[tuple[int, int]],
    min_gap_frames: int,
) -> list[tuple[int, int]]:
    """Merge consecutive runs separated by fewer than *min_gap_frames* frames.

    Parameters
    ----------
    runs : list[tuple[int, int]]
        Sorted (start, end) pairs from :func:`_find_runs`.
    min_gap_frames : int
        Runs with a gap strictly less than this are joined.

    Returns
    -------
    list[tuple[int, int]]
        Merged run list.
    """
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        gap = start - merged[-1][1] - 1
        if gap < min_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def segment_moves(
    com: pd.DataFrame,
    fps: float,
    *,
    threshold_quantile: float = 0.3,
    min_move_duration_sec: float = 0.2,
    min_rest_duration_sec: float = 0.15,
) -> list[Move]:
    """Detect discrete moves from a COM trajectory via speed thresholding.

    Parameters
    ----------
    com : pd.DataFrame
        DataFrame conforming to ``kinema.io.com.COM_SCHEMA``.
    fps : float
        Frames per second of the source video.
    threshold_quantile : float
        Quantile of the smoothed speed used as the low/high boundary.
        Frames with speed below this quantile are treated as rest.
    min_move_duration_sec : float
        Minimum duration (seconds) for a detected segment to be kept.
    min_rest_duration_sec : float
        Rest gaps shorter than this are bridged (consecutive segments merged).

    Returns
    -------
    list[Move]
        Detected moves sorted by ``start_frame``, 1-based indexed.
        Returns an empty list if no movement is detected or the trajectory
        is too short.
    """
    if len(com) < 2:
        return []

    speed = _compute_speed(com, fps)
    speed_smooth: NDArrayF64 = savgol_smooth(speed, fps, window_sec=0.5)

    threshold = float(np.nanquantile(speed_smooth, threshold_quantile))
    active: npt.NDArray[np.bool_] = speed_smooth > threshold

    runs = _find_runs(active)

    min_move_frames = max(1, int(np.ceil(min_move_duration_sec * fps)))
    runs = [(s, e) for s, e in runs if (e - s + 1) >= min_move_frames]

    min_rest_frames = max(1, int(np.ceil(min_rest_duration_sec * fps)))
    runs = _merge_runs(runs, min_gap_frames=min_rest_frames)

    # re-apply duration filter: merging can only grow runs, but be explicit
    runs = [(s, e) for s, e in runs if (e - s + 1) >= min_move_frames]

    frames = com["frame_idx"].to_numpy()
    timestamps_ms = com["timestamp_ms"].to_numpy()

    moves: list[Move] = []
    for move_idx, (pos_start, pos_end) in enumerate(runs, start=1):
        start_frame = int(frames[pos_start])
        end_frame = int(frames[pos_end])
        start_time_sec = float(timestamps_ms[pos_start]) / 1000.0
        end_time_sec = float(timestamps_ms[pos_end]) / 1000.0
        duration_sec = end_time_sec - start_time_sec

        seg_speed = speed[pos_start : pos_end + 1]
        peak_speed = float(np.nanmax(seg_speed))
        mean_speed = float(np.nanmean(seg_speed))

        nj = segment_jerk(com, pos_start, pos_end, fps)

        moves.append(
            Move(
                index=move_idx,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
                duration_sec=duration_sec,
                peak_speed=peak_speed,
                mean_speed=mean_speed,
                normalized_jerk=nj,
            )
        )

    logger.debug("Detected %d moves", len(moves))
    return moves


def write_moves(moves: list[Move], path: Path) -> None:
    """Write moves to a CSV file.

    Parameters
    ----------
    moves : list[Move]
        Output of :func:`segment_moves`.
    path : Path
        Destination CSV file path.
    """
    columns = [
        "index",
        "start_frame",
        "end_frame",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "peak_speed",
        "mean_speed",
        "normalized_jerk",
    ]
    df = pd.DataFrame([asdict(m) for m in moves]) if moves else pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


__all__ = ["Move", "segment_moves", "write_moves"]
