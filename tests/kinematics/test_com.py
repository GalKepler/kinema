"""Tests for kinema.kinematics.com and kinema.io.com."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinema.io.com import COM_SCHEMA, COMSchemaError, read_com, write_com
from kinema.io.keypoints import LANDMARK_NAMES
from kinema.kinematics.com import SEGMENT_MASS_FRACTIONS, compute_com

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_keypoints(
    positions: dict[str, tuple[float, float, float]],
    n_frames: int = 10,
    fps: float = 30.0,
    visibility: float = 1.0,
) -> pd.DataFrame:
    """Build a synthetic keypoints DataFrame with all landmarks at fixed positions.

    Parameters
    ----------
    positions : dict[str, tuple[float, float, float]]
        landmark_name → (x, y, z). Landmarks not listed get (0, 0, 0).
    n_frames : int
        Number of frames.
    fps : float
        Frames per second (used to compute timestamp_ms).
    visibility : float
        Visibility score for all landmarks.
    """
    rows = []
    for frame in range(n_frames):
        ts = int(frame * 1000.0 / fps)
        for lm_id, lm_name in enumerate(LANDMARK_NAMES):
            x, y, z = positions.get(lm_name, (0.0, 0.0, 0.0))
            rows.append(
                {
                    "frame_idx": np.int32(frame),
                    "timestamp_ms": np.int64(ts),
                    "landmark_id": np.int8(lm_id),
                    "landmark_name": lm_name,
                    "x": np.float32(x),
                    "y": np.float32(y),
                    "z": np.float32(z),
                    "visibility": np.float32(visibility),
                }
            )
    df = pd.DataFrame(rows)
    df["landmark_name"] = df["landmark_name"].astype("category")
    return df


def _uniform_positions(
    x: float = 0.5, y: float = 0.5, z: float = 0.0
) -> dict[str, tuple[float, float, float]]:
    """All landmarks at the same point → COM should equal that point."""
    return {name: (x, y, z) for name in LANDMARK_NAMES}


# ---------------------------------------------------------------------------
# SEGMENT_MASS_FRACTIONS
# ---------------------------------------------------------------------------

class TestSegmentMassFractions:
    def test_fractions_positive(self) -> None:
        assert all(v > 0 for v in SEGMENT_MASS_FRACTIONS.values())

    def test_fractions_sum_to_one(self) -> None:
        # bilateral segments appear once per side in the dict (per-side value)
        # trunk + head + 2*(upper_arm + forearm + hand + thigh + shank + foot) ≈ 1
        total = (
            SEGMENT_MASS_FRACTIONS["head"]
            + SEGMENT_MASS_FRACTIONS["trunk"]
            + 2 * SEGMENT_MASS_FRACTIONS["upper_arm"]
            + 2 * SEGMENT_MASS_FRACTIONS["forearm"]
            + 2 * SEGMENT_MASS_FRACTIONS["hand"]
            + 2 * SEGMENT_MASS_FRACTIONS["thigh"]
            + 2 * SEGMENT_MASS_FRACTIONS["shank"]
            + 2 * SEGMENT_MASS_FRACTIONS["foot"]
        )
        assert total == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_com
# ---------------------------------------------------------------------------

class TestComputeCOM:
    def test_stationary_skeleton_known_location(self) -> None:
        """All landmarks at same point → COM == that point."""
        target = (0.4, 0.6, 0.1)
        kp = _make_keypoints(_uniform_positions(*target), n_frames=15)
        result = compute_com(kp, fps=30.0)

        assert list(result.columns) == ["frame_idx", "timestamp_ms", "com_x", "com_y", "com_z"]
        assert len(result) == 15
        np.testing.assert_allclose(result["com_x"].to_numpy(), target[0], atol=1e-4)
        np.testing.assert_allclose(result["com_y"].to_numpy(), target[1], atol=1e-4)
        np.testing.assert_allclose(result["com_z"].to_numpy(), target[2], atol=1e-4)

    def test_translating_skeleton_com_tracks(self) -> None:
        """Skeleton translates linearly → COM trajectory is linear."""
        n_frames = 60
        fps = 30.0
        rows = []
        for frame in range(n_frames):
            offset_x = frame / n_frames  # 0 → 1
            ts = int(frame * 1000.0 / fps)
            for lm_id, lm_name in enumerate(LANDMARK_NAMES):
                rows.append(
                    {
                        "frame_idx": np.int32(frame),
                        "timestamp_ms": np.int64(ts),
                        "landmark_id": np.int8(lm_id),
                        "landmark_name": lm_name,
                        "x": np.float32(0.5 + offset_x),
                        "y": np.float32(0.5),
                        "z": np.float32(0.0),
                        "visibility": np.float32(1.0),
                    }
                )
        kp = pd.DataFrame(rows)
        kp["landmark_name"] = kp["landmark_name"].astype("category")

        result = compute_com(kp, fps=fps)
        com_x = result["com_x"].to_numpy(dtype=np.float64)

        # COM x should increase monotonically (smoothed linear should still be monotone)
        diffs = np.diff(com_x)
        assert np.all(diffs >= -1e-4), "COM x should be non-decreasing for rightward translation"

        # first and last frames (ignoring edge smoothing artefacts): within 5% of expected
        assert com_x[5] == pytest.approx(0.5 + 5 / n_frames, abs=0.05)
        assert com_x[-6] == pytest.approx(0.5 + (n_frames - 6) / n_frames, abs=0.05)

    def test_face_nan_does_not_corrupt_com(self) -> None:
        """NaN on face landmarks (nose, eyes, ears) → COM still finite and close to expected."""
        # build custom df with NaN x/y/z for face landmarks
        n_frames = 20
        face_landmarks = {
            "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
            "MOUTH_LEFT", "MOUTH_RIGHT",
        }
        rows = []
        fps = 30.0
        for frame in range(n_frames):
            ts = int(frame * 1000.0 / fps)
            for lm_id, lm_name in enumerate(LANDMARK_NAMES):
                if lm_name in face_landmarks:
                    x, y, z = float("nan"), float("nan"), float("nan")
                else:
                    x, y, z = 0.5, 0.5, 0.0
                rows.append(
                    {
                        "frame_idx": np.int32(frame),
                        "timestamp_ms": np.int64(ts),
                        "landmark_id": np.int8(lm_id),
                        "landmark_name": lm_name,
                        "x": np.float32(x),
                        "y": np.float32(y),
                        "z": np.float32(z),
                        "visibility": np.float32(0.0 if lm_name in face_landmarks else 1.0),
                    }
                )
        kp = pd.DataFrame(rows)
        kp["landmark_name"] = kp["landmark_name"].astype("category")

        result = compute_com(kp, fps=fps)
        assert result["com_x"].notna().all(), (
            "COM should not be NaN when only face landmarks missing"
        )
        np.testing.assert_allclose(result["com_x"].to_numpy(), 0.5, atol=0.05)

    def test_output_dtypes(self) -> None:
        kp = _make_keypoints(_uniform_positions())
        result = compute_com(kp, fps=30.0)
        assert result["frame_idx"].dtype == np.dtype("int32")
        assert result["timestamp_ms"].dtype == np.dtype("int64")
        assert result["com_x"].dtype == np.dtype("float32")
        assert result["com_y"].dtype == np.dtype("float32")
        assert result["com_z"].dtype == np.dtype("float32")


# ---------------------------------------------------------------------------
# I/O round-trip
# ---------------------------------------------------------------------------

class TestCOMIO:
    def test_roundtrip(self, tmp_path: Path) -> None:
        kp = _make_keypoints(_uniform_positions(0.3, 0.7, 0.05))
        com = compute_com(kp, fps=30.0)
        out = tmp_path / "com.parquet"
        write_com(com, out)
        loaded = read_com(out)

        pd.testing.assert_frame_equal(com.reset_index(drop=True), loaded.reset_index(drop=True))

    def test_write_rejects_wrong_schema(self, tmp_path: Path) -> None:
        bad = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(COMSchemaError):
            write_com(bad, tmp_path / "bad.parquet")

    def test_schema_columns(self) -> None:
        fields = {f.name for f in COM_SCHEMA}
        assert fields == {"frame_idx", "timestamp_ms", "com_x", "com_y", "com_z"}
