"""Tests for kinema.kinematics.smoothing."""

import numpy as np
import pytest

from kinema.kinematics.smoothing import interpolate_short_gaps, savgol_smooth


def _snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    signal_power = np.mean(clean**2)
    noise_power = np.mean((clean - noisy) ** 2)
    return 10.0 * np.log10(signal_power / noise_power)


class TestInterpolateShortGaps:
    def test_no_nans(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(interpolate_short_gaps(x, max_gap=2), x)

    def test_short_gap_filled(self) -> None:
        x = np.array([0.0, np.nan, 2.0, 3.0])
        result = interpolate_short_gaps(x, max_gap=1)
        assert not np.isnan(result[1])
        assert result[1] == pytest.approx(1.0)

    def test_long_gap_preserved(self) -> None:
        x = np.array([0.0, np.nan, np.nan, np.nan, 4.0])
        result = interpolate_short_gaps(x, max_gap=2)
        assert np.all(np.isnan(result[1:4]))

    def test_leading_nan_preserved(self) -> None:
        x = np.array([np.nan, 1.0, 2.0])
        result = interpolate_short_gaps(x, max_gap=5)
        assert np.isnan(result[0])

    def test_trailing_nan_preserved(self) -> None:
        x = np.array([1.0, 2.0, np.nan])
        result = interpolate_short_gaps(x, max_gap=5)
        assert np.isnan(result[2])


class TestSavgolSmooth:
    def test_snr_improves(self) -> None:
        rng = np.random.default_rng(42)
        fps = 30.0
        t = np.arange(60) / fps
        clean = np.sin(2 * np.pi * 1.5 * t).astype(np.float64)
        noisy = clean + rng.normal(0, 0.15, size=clean.shape)

        smoothed = savgol_smooth(noisy, fps=fps, window_sec=0.25)

        snr_before = _snr(clean, noisy)
        snr_after = _snr(clean, smoothed)
        assert snr_after > snr_before + 3, (
            f"SNR should improve by >3 dB; before={snr_before:.1f}, after={snr_after:.1f}"
        )

    def test_short_nan_gap_filled_in_output(self) -> None:
        fps = 30.0
        x = np.ones(60, dtype=np.float64)
        # insert a gap shorter than window_sec=0.25 → ~7 frames
        x[10:13] = np.nan  # 3-frame gap < 7 samples
        result = savgol_smooth(x, fps=fps, window_sec=0.25)
        assert not np.any(np.isnan(result[10:13])), "Short gap should be filled"

    def test_long_nan_gap_preserved(self) -> None:
        fps = 30.0
        x = np.ones(60, dtype=np.float64)
        # gap of 15 frames >> window (7 samples at 30fps, 0.25s)
        x[20:35] = np.nan
        result = savgol_smooth(x, fps=fps, window_sec=0.25)
        assert np.all(np.isnan(result[20:35])), "Long gap should remain NaN"

    def test_all_nan_returns_nan(self) -> None:
        x = np.full(20, np.nan)
        result = savgol_smooth(x, fps=30.0)
        assert np.all(np.isnan(result))

    def test_output_shape_preserved(self) -> None:
        x = np.linspace(0, 1, 90)
        result = savgol_smooth(x, fps=30.0)
        assert result.shape == x.shape
