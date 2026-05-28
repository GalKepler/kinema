"""Shared pytest fixtures."""

import struct
import subprocess
from pathlib import Path

import pytest


def _create_synthetic_mp4(path: Path) -> None:
    """Generate a 320x240 30fps 2-second video via ffmpeg lavfi.

    Creates a black background with a red rectangle that moves horizontally,
    ensuring decodable non-trivial content without committing binary files.
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=black:s=320x240:r=30",
            "-vf", "drawbox=x='mod(t*60\\,260)':y=90:w=60:h=60:color=red@1.0:t=fill",
            "-t", "2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _embed_cw90_rotation(path: Path) -> None:
    """Patch the tkhd transformation matrix to encode 90° CW rotation.

    Simulates a phone portrait video: pixel data is landscape 320x240, but
    the display matrix says rotate 90° CW for correct display. ffprobe then
    reports ``rotation: -90`` in the Display Matrix side data.

    Parameters
    ----------
    path : Path
        MP4 file created by :func:`_create_synthetic_mp4`.

    Raises
    ------
    RuntimeError
        If the identity matrix is not found (unexpected file structure).
    """
    stored_width = 320  # must match _create_synthetic_mp4 dimensions

    # Identity matrix as written by ffmpeg (9 big-endian int32 in 16.16 fixed-point)
    identity = struct.pack(
        ">9i",
        0x00010000, 0, 0,
        0, 0x00010000, 0,
        0, 0, 0x40000000,
    )
    # 90° CW display matrix: [0, 1, 0; -1, 0, 0; tx=W, ty=0, w=1]
    # ffprobe reports rotation=-90 for this matrix.
    rot_cw90 = struct.pack(
        ">9i",
        0, 0x00010000, 0,
        -0x00010000, 0, 0,
        stored_width * 0x00010000, 0, 0x40000000,
    )

    data = bytearray(path.read_bytes())
    idx = data.find(identity)
    if idx == -1:
        raise RuntimeError(f"tkhd identity matrix not found in {path}")
    data[idx : idx + 36] = rot_cw90
    path.write_bytes(bytes(data))


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-cached 320x240 30fps 2-second video, no rotation metadata."""
    out = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    _create_synthetic_mp4(out)
    return out


@pytest.fixture(scope="session")
def synthetic_video_rotated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-cached 320x240 stored, 90° CW rotation matrix → display 240x320.

    Simulates a portrait phone video where stored pixel dimensions are
    landscape 320x240 but the tkhd display matrix encodes 90° CW rotation,
    making display dimensions 240 wide x 320 tall.
    """
    out = tmp_path_factory.mktemp("video_rot") / "synthetic_rotated.mp4"
    _create_synthetic_mp4(out)
    _embed_cw90_rotation(out)
    return out
