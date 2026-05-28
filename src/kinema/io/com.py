"""Parquet schema and I/O for center-of-mass trajectory. Single source of truth for COM schema."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COM_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("frame_idx", pa.int32()),
        pa.field("timestamp_ms", pa.int64()),
        pa.field("com_x", pa.float32()),
        pa.field("com_y", pa.float32()),
        pa.field("com_z", pa.float32()),
    ]
)

_NUMERIC_DTYPES: dict[str, np.dtype[np.generic]] = {
    "frame_idx": np.dtype("int32"),
    "timestamp_ms": np.dtype("int64"),
    "com_x": np.dtype("float32"),
    "com_y": np.dtype("float32"),
    "com_z": np.dtype("float32"),
}

_REQUIRED_COLUMNS: tuple[str, ...] = ("frame_idx", "timestamp_ms", "com_x", "com_y", "com_z")


class COMSchemaError(Exception):
    """Raised when a DataFrame does not conform to COM_SCHEMA."""


def _validate(df: pd.DataFrame) -> None:
    """Validate that *df* conforms to COM_SCHEMA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    COMSchemaError
        If any column is missing or has the wrong dtype.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise COMSchemaError(f"Missing columns: {missing}")

    for col, expected in _NUMERIC_DTYPES.items():
        actual_dtype = df[col].dtype
        numpy_dtype: np.dtype[np.generic]
        if hasattr(actual_dtype, "numpy_dtype"):
            numpy_dtype = actual_dtype.numpy_dtype
        else:
            numpy_dtype = np.dtype(actual_dtype)  # type: ignore[arg-type]
        if numpy_dtype != expected:
            raise COMSchemaError(
                f"Column '{col}': expected dtype {expected}, got {numpy_dtype}"
            )


def write_com(df: pd.DataFrame, path: Path) -> None:
    """Validate and write COM DataFrame to Parquet with snappy compression.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame conforming to COM_SCHEMA.
    path : Path
        Destination Parquet file path.

    Raises
    ------
    COMSchemaError
        If *df* does not conform to COM_SCHEMA.
    """
    _validate(df)
    table = pa.Table.from_pandas(df, schema=COM_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def read_com(path: Path) -> pd.DataFrame:
    """Read a COM Parquet file and return a validated DataFrame.

    Parameters
    ----------
    path : Path
        Path to the Parquet file written by :func:`write_com`.

    Returns
    -------
    pd.DataFrame
        DataFrame with dtypes matching COM_SCHEMA.

    Raises
    ------
    COMSchemaError
        If the file does not conform to COM_SCHEMA.
    """
    table = pq.read_table(path, schema=COM_SCHEMA)
    df: pd.DataFrame = table.to_pandas()
    _validate(df)
    return df
