"""Parquet schema and I/O for keypoint time-series. Single source of truth for schema."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LANDMARK_NAMES: tuple[str, ...] = (
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

KEYPOINTS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("frame_idx", pa.int32()),
        pa.field("timestamp_ms", pa.int64()),
        pa.field("landmark_id", pa.int8()),
        pa.field("landmark_name", pa.dictionary(pa.int8(), pa.utf8())),
        pa.field("x", pa.float32()),
        pa.field("y", pa.float32()),
        pa.field("z", pa.float32()),
        pa.field("visibility", pa.float32()),
    ]
)

# Exact numpy dtypes required for numeric columns.
_NUMERIC_DTYPES: dict[str, np.dtype[np.generic]] = {
    "frame_idx": np.dtype("int32"),
    "timestamp_ms": np.dtype("int64"),
    "landmark_id": np.dtype("int8"),
    "x": np.dtype("float32"),
    "y": np.dtype("float32"),
    "z": np.dtype("float32"),
    "visibility": np.dtype("float32"),
}

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "frame_idx",
    "timestamp_ms",
    "landmark_id",
    "landmark_name",
    "x",
    "y",
    "z",
    "visibility",
)


class SchemaError(Exception):
    """Raised when a DataFrame does not conform to KEYPOINTS_SCHEMA."""


def _validate(df: pd.DataFrame) -> None:
    """Validate that *df* conforms to KEYPOINTS_SCHEMA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    SchemaError
        If any column is missing, has the wrong dtype, or contains out-of-range values.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing columns: {missing}")

    for col, expected in _NUMERIC_DTYPES.items():
        actual_dtype = df[col].dtype
        # Unwrap pandas ExtensionDtype (e.g. ArrowDtype) to its numpy equivalent.
        numpy_dtype: np.dtype[np.generic]
        if hasattr(actual_dtype, "numpy_dtype"):
            numpy_dtype = actual_dtype.numpy_dtype
        else:
            numpy_dtype = np.dtype(actual_dtype)  # type: ignore[arg-type]
        if numpy_dtype != expected:
            raise SchemaError(
                f"Column '{col}': expected dtype {expected}, got {numpy_dtype}"
            )

    name_dtype = df["landmark_name"].dtype
    is_acceptable_name_dtype = isinstance(
        name_dtype, (pd.StringDtype, pd.CategoricalDtype)
    ) or str(name_dtype) == "object"
    if not is_acceptable_name_dtype:
        raise SchemaError(
            f"Column 'landmark_name': expected string or categorical dtype, got {name_dtype}"
        )

    vis = df["visibility"]
    if (vis < 0.0).any() or (vis > 1.0).any():
        raise SchemaError("Column 'visibility': values must be in [0, 1]")

    lid = df["landmark_id"]
    if (lid < 0).any() or (lid > 32).any():
        raise SchemaError("Column 'landmark_id': values must be in [0, 32]")


def write_keypoints(df: pd.DataFrame, path: Path) -> None:
    """Validate and write keypoints DataFrame to Parquet with snappy compression.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame conforming to KEYPOINTS_SCHEMA.
    path : Path
        Destination Parquet file path.

    Raises
    ------
    SchemaError
        If *df* does not conform to KEYPOINTS_SCHEMA.
    """
    _validate(df)
    table = pa.Table.from_pandas(df, schema=KEYPOINTS_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="snappy")  # type: ignore[no-untyped-call]


def read_keypoints(path: Path) -> pd.DataFrame:
    """Read a keypoints Parquet file and return a validated DataFrame.

    Parameters
    ----------
    path : Path
        Path to the Parquet file written by :func:`write_keypoints`.

    Returns
    -------
    pd.DataFrame
        DataFrame with dtypes matching KEYPOINTS_SCHEMA. ``landmark_name``
        is returned as a pandas Categorical column.

    Raises
    ------
    SchemaError
        If the file does not conform to KEYPOINTS_SCHEMA.
    """
    table = pq.read_table(path, schema=KEYPOINTS_SCHEMA)  # type: ignore[no-untyped-call]
    df: pd.DataFrame = table.to_pandas(categories=["landmark_name"])
    _validate(df)
    return df
