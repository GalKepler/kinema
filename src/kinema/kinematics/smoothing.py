"""Signal smoothing utilities for kinematic time-series."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

NDArrayF = npt.NDArray[np.floating[Any]]


def interpolate_short_gaps(x: NDArrayF, max_gap: int) -> NDArrayF:
    """Linearly interpolate NaN runs up to *max_gap* samples long.

    Parameters
    ----------
    x : NDArrayF
        1-D float array, may contain NaN.
    max_gap : int
        Maximum consecutive NaN count to fill. Longer runs are left as NaN.

    Returns
    -------
    NDArrayF
        Copy of *x* with short NaN gaps filled by linear interpolation.
    """
    if max_gap <= 0:
        return x.copy()

    out: NDArrayF = x.copy()
    n = len(out)
    i = 0
    while i < n:
        if not np.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(out[j]):
            j += 1
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and j < n:
            left = out[i - 1]
            right = out[j]
            for k in range(gap_len):
                out[i + k] = left + (right - left) * (k + 1) / (gap_len + 1)
        i = j
    return out


def savgol_smooth(
    x: NDArrayF,
    fps: float,
    window_sec: float = 0.25,
    polyorder: int = 3,
) -> NDArrayF:
    """Savitzky-Golay smooth a 1-D signal, handling NaN gaps gracefully.

    Short NaN gaps (< *window_sec*) are interpolated before smoothing and the
    interpolated values are retained.  Long NaN gaps (>= *window_sec*) are
    interpolated for the purpose of smoothing only, then restored as NaN in
    the output so callers know data was missing.

    Parameters
    ----------
    x : NDArrayF
        1-D float array.
    fps : float
        Frames per second of the signal.
    window_sec : float
        Smoothing window length in seconds.  Converted to the nearest odd
        integer number of samples; minimum 3.
    polyorder : int
        Polynomial order for the Savitzky-Golay filter.  Must be less than
        the window length.

    Returns
    -------
    NDArrayF
        Smoothed array of the same shape as *x*.  NaN positions that
        correspond to long gaps are restored.
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")

    window_samples = round(window_sec * fps)
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = max(window_samples, polyorder + 1)
    if window_samples % 2 == 0:
        window_samples += 1

    short_gap = window_samples - 1
    nan_mask = np.isnan(x)

    filled = interpolate_short_gaps(x, max_gap=short_gap)
    long_nan_mask = np.isnan(filled)

    if long_nan_mask.any():
        filled = interpolate_short_gaps(filled, max_gap=len(filled))
        if np.isnan(filled).any():
            idx = np.where(~np.isnan(filled))[0]
            if len(idx) == 0:
                return x.copy()
            filled = np.interp(np.arange(len(filled)), idx, filled[idx])

    smoothed: NDArrayF = savgol_filter(filled, window_length=window_samples, polyorder=polyorder)

    result: NDArrayF = smoothed.copy()
    result[long_nan_mask] = np.nan
    result[nan_mask & long_nan_mask] = np.nan
    return result
