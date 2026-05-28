"""Tests for kinema.io.video."""

from pathlib import Path

import numpy as np
import pytest

from kinema.io.video import VideoMetadata, iter_frames, probe_video


def test_probe_video_returns_sensible_values(synthetic_video: Path) -> None:
    meta = probe_video(synthetic_video)

    assert isinstance(meta, VideoMetadata)
    assert meta.fps == pytest.approx(30.0, rel=0.01)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.duration_sec == pytest.approx(2.0, abs=0.1)
    assert meta.frame_count > 0
    assert meta.rotation_degrees == 0
    assert meta.path == synthetic_video


def test_iter_frames_count_dtype_shape(synthetic_video: Path) -> None:
    meta = probe_video(synthetic_video)
    frames = list(iter_frames(synthetic_video))

    # Allow ±1 frame: rounding difference between ffprobe nb_frames and OpenCV
    assert abs(len(frames) - meta.frame_count) <= 1

    frame_idx, timestamp_sec, frame = frames[0]
    assert frame_idx == 0
    assert timestamp_sec == pytest.approx(0.0)
    assert frame.dtype == np.uint8
    assert frame.shape == (meta.height, meta.width, 3)


def test_rotation_applied(synthetic_video_rotated: Path) -> None:
    """Stored 320x240 + 90° CW display matrix → display 240x320."""
    meta = probe_video(synthetic_video_rotated)

    assert meta.rotation_degrees == 90
    # Display dimensions swap: stored 320x240 → display 240 wide x 320 tall
    assert meta.width == 240
    assert meta.height == 320

    _, _, frame = next(iter(iter_frames(synthetic_video_rotated)))
    assert frame.dtype == np.uint8
    # numpy shape is (H, W, C) = (display_height, display_width, 3)
    assert frame.shape == (meta.height, meta.width, 3)
