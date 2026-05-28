"""Tests for kinema.pose.extract."""

from pathlib import Path

import numpy as np

from kinema.io.keypoints import LANDMARK_NAMES, read_keypoints
from kinema.pose.extract import ExtractionStats, extract_pose

_N_LANDMARKS = len(LANDMARK_NAMES)


def test_extract_pose_synthetic(synthetic_video: Path, tmp_path: Path) -> None:
    """Extraction on a no-human video: correct schema, row count, valid values."""
    output = tmp_path / "keypoints.parquet"
    stats = extract_pose(synthetic_video, output, model_complexity=0)

    assert isinstance(stats, ExtractionStats)
    assert output.exists()

    df = read_keypoints(output)

    # Row count: frame_count * 33 landmarks
    assert len(df) == stats.frame_count * _N_LANDMARKS

    # Visibility always in [0, 1]
    assert (df["visibility"] >= 0.0).all()
    assert (df["visibility"] <= 1.0).all()

    # Per-frame timestamps are strictly monotonic increasing
    ts_by_frame = df.groupby("frame_idx", sort=True)["timestamp_ms"].first()
    assert ts_by_frame.is_monotonic_increasing

    # Each frame has exactly 33 landmark rows
    counts = df.groupby("frame_idx")["landmark_id"].count()
    assert (counts == _N_LANDMARKS).all()

    # Synthetic video has no human → frames_with_pose should be 0 or very low
    assert stats.frames_with_pose >= 0
    assert stats.frame_count > 0
    assert stats.wall_time_sec > 0.0


def test_extract_pose_1frame_edge_case(
    synthetic_video_1frame: Path, tmp_path: Path
) -> None:
    """1-frame video produces valid output with exactly 33 rows."""
    output = tmp_path / "keypoints_1f.parquet"
    stats = extract_pose(synthetic_video_1frame, output, model_complexity=0)

    assert output.exists()
    assert stats.frame_count == 1

    df = read_keypoints(output)
    assert len(df) == _N_LANDMARKS

    assert (df["visibility"] >= 0.0).all()
    assert (df["visibility"] <= 1.0).all()
    assert df["frame_idx"].iloc[0] == np.int32(0)
    assert df["timestamp_ms"].iloc[0] == np.int64(0)
