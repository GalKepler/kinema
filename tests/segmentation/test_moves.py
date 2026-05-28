"""Tests for kinema.segmentation.moves."""

from pathlib import Path

import numpy as np
import pandas as pd

from kinema.segmentation.moves import segment_moves, write_moves

FPS = 30.0


def _make_com(
    total_sec: float,
    fps: float,
    move_intervals: list[tuple[float, float]],
    amplitude: float = 0.05,
) -> pd.DataFrame:
    """Build a COM DataFrame with sinusoidal motion during move intervals.

    During *move_intervals* the x-coordinate follows a sine wave (non-zero
    velocity); outside intervals x is constant (velocity = 0).

    Parameters
    ----------
    total_sec : float
        Total trajectory duration.
    fps : float
        Frames per second.
    move_intervals : list of (start_sec, end_sec) pairs
        Time ranges with movement.
    amplitude : float
        Sine amplitude in normalized units.
    """
    n = int(total_sec * fps)
    t = np.arange(n, dtype=np.float64) / fps
    x = np.zeros(n, dtype=np.float64)

    for start, end in move_intervals:
        mask = (t >= start) & (t < end)
        duration = end - start
        # 3 full cycles over the move to ensure measurable velocity
        x[mask] = amplitude * np.sin(2 * np.pi * 3.0 * (t[mask] - start) / duration)

    frames = np.arange(n, dtype=np.int32)
    timestamps_ms = (frames * 1000.0 / fps).astype(np.int64)

    return pd.DataFrame(
        {
            "frame_idx": frames,
            "timestamp_ms": timestamps_ms,
            "com_x": x.astype(np.float32),
            "com_y": np.zeros(n, dtype=np.float32),
            "com_z": np.zeros(n, dtype=np.float32),
        }
    )


def test_three_moves_detected() -> None:
    """Trajectory with 3 clear move/rest cycles yields exactly 3 moves."""
    # rest=2.0s >> 0.5s smoothing window; move=1.0s >> 0.2s min_move_duration
    com = _make_com(
        total_sec=12.0,
        fps=FPS,
        move_intervals=[(2.0, 3.0), (5.0, 6.0), (8.0, 9.0)],
    )
    moves = segment_moves(com, FPS)
    assert len(moves) == 3
    for i, m in enumerate(moves, start=1):
        assert m.index == i
        assert m.start_frame < m.end_frame
        assert m.duration_sec > 0.0
        assert m.peak_speed > 0.0
        assert m.mean_speed > 0.0


def test_no_movement_returns_empty() -> None:
    """Flat COM trajectory (zero velocity) yields no moves."""
    n = int(FPS * 5)
    frames = np.arange(n, dtype=np.int32)
    timestamps_ms = (frames * 1000.0 / FPS).astype(np.int64)
    com = pd.DataFrame(
        {
            "frame_idx": frames,
            "timestamp_ms": timestamps_ms,
            "com_x": np.zeros(n, dtype=np.float32),
            "com_y": np.zeros(n, dtype=np.float32),
            "com_z": np.zeros(n, dtype=np.float32),
        }
    )
    moves = segment_moves(com, FPS)
    assert moves == []


def test_sub_duration_moves_filtered() -> None:
    """Moves shorter than min_move_duration_sec are dropped.

    The 0.5 s speed-smoothing window spreads a 0.1 s impulse over ~0.5 s
    (≈15 frames at 30 fps).  With min_move_duration_sec=0.8 s (24 frames),
    the blurred short event is filtered while the 2.0 s long move survives.
    """
    com = _make_com(
        total_sec=14.0,
        fps=FPS,
        move_intervals=[(1.0, 3.0), (8.0, 8.1)],
    )
    moves = segment_moves(com, FPS, min_move_duration_sec=0.8)
    assert len(moves) == 1
    assert moves[0].index == 1


def test_short_rest_gap_merges_moves() -> None:
    """Two moves separated by a rest < min_rest_duration_sec are merged into one."""
    # gap = 0.05s < default min_rest_duration_sec = 0.15s → merged
    com = _make_com(
        total_sec=10.0,
        fps=FPS,
        move_intervals=[(2.0, 3.0), (3.05, 4.05)],
    )
    moves = segment_moves(com, FPS, min_rest_duration_sec=0.15)
    assert len(moves) == 1


def test_csv_roundtrip(tmp_path: Path) -> None:
    """write_moves produces a CSV that can be read back with correct shape and values."""
    com = _make_com(
        total_sec=12.0,
        fps=FPS,
        move_intervals=[(2.0, 3.0), (5.0, 6.0), (8.0, 9.0)],
    )
    moves = segment_moves(com, FPS)
    assert len(moves) > 0

    path = tmp_path / "moves.csv"
    write_moves(moves, path)

    df = pd.read_csv(path)
    assert len(df) == len(moves)
    expected_columns = [
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
    assert list(df.columns) == expected_columns
    assert df["index"].tolist() == [m.index for m in moves]
    assert df["start_frame"].tolist() == [m.start_frame for m in moves]
    assert df["end_frame"].tolist() == [m.end_frame for m in moves]


def test_write_moves_empty_csv(tmp_path: Path) -> None:
    """write_moves with an empty list writes a valid CSV with headers only."""
    path = tmp_path / "empty.csv"
    write_moves([], path)

    df = pd.read_csv(path)
    assert len(df) == 0
    assert "index" in df.columns
    assert "normalized_jerk" in df.columns
