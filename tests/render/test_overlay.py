"""Tests for kinema.render.overlay."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from kinema.io.com import write_com
from kinema.io.keypoints import LANDMARK_NAMES, write_keypoints
from kinema.io.video import probe_video
from kinema.render.overlay import POSE_CONNECTIONS, render_overlay


def _write_empty_keypoints(tmp_path: Path) -> Path:
    """Write a zero-row keypoints Parquet."""
    df = pd.DataFrame(
        {
            "frame_idx": pd.array([], dtype="int32"),
            "timestamp_ms": pd.array([], dtype="int64"),
            "landmark_id": pd.array([], dtype="int8"),
            "landmark_name": pd.Categorical([], categories=list(LANDMARK_NAMES)),
            "x": pd.array([], dtype="float32"),
            "y": pd.array([], dtype="float32"),
            "z": pd.array([], dtype="float32"),
            "visibility": pd.array([], dtype="float32"),
        }
    )
    path = tmp_path / "keypoints.parquet"
    write_keypoints(df, path)
    return path


def _write_synthetic_com(tmp_path: Path, frame_count: int, fps: float) -> Path:
    """Write COM parquet with COM at (0.5, 0.5) for every frame."""
    frames = np.arange(frame_count, dtype="int32")
    timestamps = (frames * 1000.0 / fps).astype("int64")
    df = pd.DataFrame(
        {
            "frame_idx": frames,
            "timestamp_ms": timestamps,
            "com_x": np.full(frame_count, 0.5, dtype="float32"),
            "com_y": np.full(frame_count, 0.5, dtype="float32"),
            "com_z": np.zeros(frame_count, dtype="float32"),
        }
    )
    path = tmp_path / "com.parquet"
    write_com(df, path)
    return path


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        n += 1
    cap.release()
    return n


class TestRenderOverlay:
    def test_no_com_produces_valid_video(self, synthetic_video: Path, tmp_path: Path) -> None:
        meta = probe_video(synthetic_video)
        kp_path = _write_empty_keypoints(tmp_path)
        out = tmp_path / "overlay.mp4"

        render_overlay(synthetic_video, kp_path, out)

        assert out.exists(), "Output file not created"
        cap = cv2.VideoCapture(str(out))
        assert cap.isOpened(), "OpenCV cannot open output video"
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert out_w == meta.width
        assert out_h == meta.height
        assert _count_frames(out) == meta.frame_count

    def test_with_com_produces_valid_video(self, synthetic_video: Path, tmp_path: Path) -> None:
        meta = probe_video(synthetic_video)
        kp_path = _write_empty_keypoints(tmp_path)
        com_path = _write_synthetic_com(tmp_path, meta.frame_count, meta.fps)
        out = tmp_path / "overlay_com.mp4"

        render_overlay(synthetic_video, kp_path, out, com_path=com_path)

        assert out.exists()
        assert _count_frames(out) == meta.frame_count

    def test_pose_connections_valid_landmark_ids(self) -> None:
        n_landmarks = len(LANDMARK_NAMES)
        for a, b in POSE_CONNECTIONS:
            assert 0 <= a < n_landmarks, f"Connection endpoint {a} out of range"
            assert 0 <= b < n_landmarks, f"Connection endpoint {b} out of range"
