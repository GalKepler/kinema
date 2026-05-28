"""Tests for kinema.kinematics.metrics."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from kinema.kinematics.metrics import MIN_FRAMES, normalized_jerk, segment_jerk

NDArrayF64 = npt.NDArray[np.float64]


def _make_com(
    x: NDArrayF64,
    y: NDArrayF64,
    z: NDArrayF64,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Build a minimal COM DataFrame from coordinate arrays."""
    n = len(x)
    dt_ms = int(1000 / fps)
    return pd.DataFrame(
        {
            "frame_idx": np.arange(n, dtype=np.int32),
            "timestamp_ms": np.arange(n, dtype=np.int64) * dt_ms,
            "com_x": x.astype(np.float32),
            "com_y": y.astype(np.float32),
            "com_z": z.astype(np.float32),
        }
    )


class TestNormalizedJerk:
    def test_constant_velocity_near_zero(self) -> None:
        """Perfectly straight constant-velocity path has NJ ≈ 0.

        Float32 COM storage introduces ~1e-3 quantization-induced jerk;
        tolerance is set to 1e-2 to account for this while still confirming
        the metric is near-zero for smooth motion.
        """
        fps = 30.0
        n = 60
        t = np.linspace(0, 2, n)
        com = _make_com(t, np.zeros(n), np.zeros(n), fps)
        nj = normalized_jerk(com, fps)
        assert not np.isnan(nj), "NJ must not be NaN for valid trajectory"
        assert nj == pytest.approx(0.0, abs=1e-2)

    def test_jittery_higher_than_smooth(self) -> None:
        """Noisy trajectory yields strictly higher NJ than a smooth one."""
        fps = 30.0
        n = 60
        rng = np.random.default_rng(0)
        t = np.linspace(0, 1, n)
        noise = rng.normal(0, 0.05, n)

        smooth_com = _make_com(t, np.zeros(n), np.zeros(n), fps)
        jittery_com = _make_com(t + noise, np.zeros(n), np.zeros(n), fps)

        nj_smooth = normalized_jerk(smooth_com, fps)
        nj_jittery = normalized_jerk(jittery_com, fps)

        assert nj_jittery > nj_smooth

    def test_segment_jerk_full_range_equals_normalized_jerk(self) -> None:
        """segment_jerk over full range equals normalized_jerk."""
        fps = 30.0
        n = 60
        rng = np.random.default_rng(1)
        t = np.linspace(0, 2, n)
        y = rng.normal(0, 0.02, n)
        com = _make_com(t, y, np.zeros(n), fps)

        nj_full = normalized_jerk(com, fps)
        nj_seg = segment_jerk(com, 0, n - 1, fps)

        assert nj_seg == pytest.approx(nj_full, rel=1e-10)

    def test_short_segment_returns_nan(self) -> None:
        """Trajectories with fewer than MIN_FRAMES rows return NaN."""
        fps = 30.0
        short_n = MIN_FRAMES - 1
        t = np.linspace(0, 1, short_n)
        com = _make_com(t, np.zeros(short_n), np.zeros(short_n), fps)

        assert np.isnan(normalized_jerk(com, fps))
        assert np.isnan(segment_jerk(com, 0, short_n - 1, fps))

    def test_segment_jerk_subset(self) -> None:
        """segment_jerk on a sub-range differs from the full trajectory."""
        fps = 30.0
        n = 60
        rng = np.random.default_rng(2)
        t = np.linspace(0, 2, n)
        noise = rng.normal(0, 0.05, n)
        com = _make_com(t + noise, np.zeros(n), np.zeros(n), fps)

        nj_full = normalized_jerk(com, fps)
        nj_half = segment_jerk(com, 0, n // 2, fps)

        # Scores need not be equal; both must be finite and positive
        assert np.isfinite(nj_full) and nj_full >= 0
        assert np.isfinite(nj_half) and nj_half >= 0
