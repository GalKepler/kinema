"""Tests for kinema.io.keypoints."""

import numpy as np
import pandas as pd
import pytest

from kinema.io.keypoints import (
    KEYPOINTS_SCHEMA,
    LANDMARK_NAMES,
    SchemaError,
    read_keypoints,
    write_keypoints,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MEDIAPIPE_POSE_LANDMARK_NAMES: tuple[str, ...] = (
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)

_N_LANDMARKS = 33
_N_FRAMES = 5


def _make_valid_df(
    n_frames: int = _N_FRAMES,
    n_landmarks: int = _N_LANDMARKS,
) -> pd.DataFrame:
    """Build a minimal valid keypoints DataFrame."""
    frame_idxs = np.repeat(np.arange(n_frames, dtype=np.int32), n_landmarks)
    landmark_ids = np.tile(np.arange(n_landmarks, dtype=np.int8), n_frames)
    n_rows = n_frames * n_landmarks
    return pd.DataFrame(
        {
            "frame_idx": frame_idxs,
            "timestamp_ms": (frame_idxs * 33).astype(np.int64),
            "landmark_id": landmark_ids,
            "landmark_name": [LANDMARK_NAMES[i] for i in landmark_ids],
            "x": np.random.default_rng(0).random(n_rows).astype(np.float32),
            "y": np.random.default_rng(1).random(n_rows).astype(np.float32),
            "z": np.random.default_rng(2).random(n_rows).astype(np.float32) - 0.5,
            "visibility": np.random.default_rng(3).random(n_rows).astype(np.float32),
        }
    )


# ---------------------------------------------------------------------------
# LANDMARK_NAMES
# ---------------------------------------------------------------------------


def test_landmark_names_count() -> None:
    assert len(LANDMARK_NAMES) == 33


def test_landmark_names_match_mediapipe_docs() -> None:
    assert LANDMARK_NAMES == _MEDIAPIPE_POSE_LANDMARK_NAMES


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_valid_df()
    path = tmp_path / "keypoints.parquet"  # type: ignore[operator]
    write_keypoints(df, path)
    result = read_keypoints(path)

    # Convert landmark_name to plain str in both: category sort order can differ
    # (pandas astype → alpha, pyarrow dict → insertion order).
    df_norm = df.assign(landmark_name=df["landmark_name"].astype(str))
    result_norm = result.assign(landmark_name=result["landmark_name"].astype(str))
    pd.testing.assert_frame_equal(df_norm, result_norm)


# ---------------------------------------------------------------------------
# Schema rejection tests
# ---------------------------------------------------------------------------


def test_reject_missing_column_x(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_valid_df().drop(columns=["x"])
    with pytest.raises(SchemaError, match="Missing columns"):
        write_keypoints(df, tmp_path / "kp.parquet")  # type: ignore[operator]


def test_reject_missing_column_visibility(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_valid_df().drop(columns=["visibility"])
    with pytest.raises(SchemaError, match="Missing columns"):
        write_keypoints(df, tmp_path / "kp.parquet")  # type: ignore[operator]


def test_reject_wrong_dtype_x(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_valid_df()
    df["x"] = df["x"].astype(np.float64)
    with pytest.raises(SchemaError, match="'x'"):
        write_keypoints(df, tmp_path / "kp.parquet")  # type: ignore[operator]


def test_reject_visibility_above_one(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_valid_df()
    df.loc[0, "visibility"] = np.float32(1.1)
    with pytest.raises(SchemaError, match="visibility"):
        write_keypoints(df, tmp_path / "kp.parquet")  # type: ignore[operator]


def test_reject_visibility_below_zero(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_valid_df()
    df.loc[0, "visibility"] = np.float32(-0.01)
    with pytest.raises(SchemaError, match="visibility"):
        write_keypoints(df, tmp_path / "kp.parquet")  # type: ignore[operator]


def test_reject_landmark_id_above_32(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_valid_df()
    df.loc[0, "landmark_id"] = np.int8(33)
    with pytest.raises(SchemaError, match="landmark_id"):
        write_keypoints(df, tmp_path / "kp.parquet")  # type: ignore[operator]


def test_reject_landmark_id_below_zero(tmp_path: pytest.TempPathFactory) -> None:
    df = _make_valid_df()
    df.loc[0, "landmark_id"] = np.int8(-1)
    with pytest.raises(SchemaError, match="landmark_id"):
        write_keypoints(df, tmp_path / "kp.parquet")  # type: ignore[operator]


# ---------------------------------------------------------------------------
# KEYPOINTS_SCHEMA sanity
# ---------------------------------------------------------------------------


def test_schema_has_eight_fields() -> None:
    assert len(KEYPOINTS_SCHEMA) == 8


def test_schema_field_names() -> None:
    names = KEYPOINTS_SCHEMA.names
    assert names == [
        "frame_idx",
        "timestamp_ms",
        "landmark_id",
        "landmark_name",
        "x",
        "y",
        "z",
        "visibility",
    ]
