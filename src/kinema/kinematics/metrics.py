"""Movement metrics: normalized jerk, smoothness.

Normalized jerk (NJ) quantifies trajectory smoothness via the third
derivative of position.  **Lower NJ = smoother movement.**

Formula (Balasubramanian et al. 2015):

    NJ = sqrt(0.5 · ∫ ‖j(t)‖² dt · T⁵ / L²)

where j(t) is the 3-D jerk vector (third derivative of COM position),
T is total duration in seconds, and L is total arc-length.
Derivatives are computed by three successive applications of
``numpy.gradient`` (second-order central finite differences) on the
already-smoothed COM trajectory.

References
----------
Balasubramanian S, Melendez-Calderon A, Roby-Brami A, Burdet E (2015).
"On the analysis of movement smoothness."
J NeuroEngineering Rehabil 12:103. https://doi.org/10.1186/s12984-015-0090-9
"""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import trapezoid as scipy_trapezoid

logger = logging.getLogger(__name__)

NDArrayF64 = npt.NDArray[np.float64]

MIN_FRAMES: int = 5
"""Minimum frame count required to compute a meaningful NJ value.

Segments shorter than this return NaN.  Five frames are needed so that
three successive gradient passes still have interior support at the
boundary samples.
"""


def _normalized_jerk_from_arrays(
    x: NDArrayF64,
    y: NDArrayF64,
    z: NDArrayF64,
    fps: float,
) -> float:
    """Compute NJ from raw coordinate arrays.

    Parameters
    ----------
    x, y, z : NDArrayF64
        1-D position arrays (same length, already smoothed).
    fps : float
        Frames per second.

    Returns
    -------
    float
        Dimensionless normalized jerk, or NaN if ``len(x) < MIN_FRAMES``.
    """
    n = len(x)
    if n < MIN_FRAMES:
        return float("nan")

    dt = 1.0 / fps

    def _jerk_1d(arr: NDArrayF64) -> NDArrayF64:
        d1: NDArrayF64 = np.gradient(arr, dt)
        d2: NDArrayF64 = np.gradient(d1, dt)
        d3: NDArrayF64 = np.gradient(d2, dt)
        return d3

    jx = _jerk_1d(x)
    jy = _jerk_1d(y)
    jz = _jerk_1d(z)

    j_sq: NDArrayF64 = jx**2 + jy**2 + jz**2
    integral_j2 = float(scipy_trapezoid(j_sq, dx=dt))

    duration = (n - 1) * dt

    seg_x = np.diff(x)
    seg_y = np.diff(y)
    seg_z = np.diff(z)
    path_length = float(np.sum(np.sqrt(seg_x**2 + seg_y**2 + seg_z**2)))

    if path_length == 0.0:
        # Stationary: jerk is zero → NJ = 0 by definition.
        return 0.0

    return float(np.sqrt(0.5 * integral_j2 * duration**5 / path_length**2))


def normalized_jerk(com: pd.DataFrame, fps: float) -> float:
    """Compute dimensionless normalized jerk for a full COM trajectory.

    Parameters
    ----------
    com : pd.DataFrame
        DataFrame conforming to ``kinema.io.com.COM_SCHEMA``.
        Must contain columns ``com_x``, ``com_y``, ``com_z``
        (already smoothed normalized coordinates).
    fps : float
        Frames per second of the source video.

    Returns
    -------
    float
        Normalized jerk score (dimensionless, lower = smoother).
        Returns ``float('nan')`` when the trajectory has fewer than
        ``MIN_FRAMES`` rows.

    Notes
    -----
    Jerk is the third time-derivative of position, computed here via
    three successive applications of ``numpy.gradient`` (second-order
    central finite differences).  Integration uses the trapezoidal rule.
    """
    x = com["com_x"].to_numpy(dtype=np.float64)
    y = com["com_y"].to_numpy(dtype=np.float64)
    z = com["com_z"].to_numpy(dtype=np.float64)
    return _normalized_jerk_from_arrays(x, y, z, fps)


def segment_jerk(
    com: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    fps: float,
) -> float:
    """Compute normalized jerk over a positional index range of a COM trajectory.

    Parameters
    ----------
    com : pd.DataFrame
        Full COM trajectory (``kinema.io.com.COM_SCHEMA``).
    start_idx : int
        First row to include (positional, ``iloc``-style, inclusive).
    end_idx : int
        Last row to include (positional, ``iloc``-style, inclusive).
    fps : float
        Frames per second of the source video.

    Returns
    -------
    float
        Normalized jerk score for the slice, or ``float('nan')`` if the
        slice contains fewer than ``MIN_FRAMES`` rows.
    """
    segment = com.iloc[start_idx : end_idx + 1]
    x = segment["com_x"].to_numpy(dtype=np.float64)
    y = segment["com_y"].to_numpy(dtype=np.float64)
    z = segment["com_z"].to_numpy(dtype=np.float64)
    return _normalized_jerk_from_arrays(x, y, z, fps)


__all__ = [
    "MIN_FRAMES",
    "normalized_jerk",
    "segment_jerk",
]
