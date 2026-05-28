"""Video I/O utilities using OpenCV.

Requires ``ffprobe`` and ``ffmpeg`` on PATH. See README.md for system
dependency installation instructions.
"""

import json
import logging
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maps rotation_degrees → OpenCV rotate code that corrects stored frames
# to display orientation. rotate=90 means stored frame needs CCW 90° to
# display correctly (phone held portrait; sensor landscape).
_ROTATE_MAP: dict[int, int] = {
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata for a video file with display-correct dimensions.

    Parameters
    ----------
    path : Path
        Absolute path to the video file.
    width : int
        Display width in pixels (post-rotation).
    height : int
        Display height in pixels (post-rotation).
    fps : float
        Frames per second.
    frame_count : int
        Total number of video frames.
    duration_sec : float
        Duration in seconds.
    rotation_degrees : int
        CW rotation stored in file metadata (0, 90, 180, or 270).
        Callers never need to act on this; ``iter_frames`` applies it.
    """

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    rotation_degrees: int


def probe_video(path: Path) -> VideoMetadata:
    """Read video metadata via ffprobe.

    Parameters
    ----------
    path : Path
        Path to the video file.

    Returns
    -------
    VideoMetadata
        Metadata with display-correct (post-rotation) ``width`` and ``height``.

    Raises
    ------
    FileNotFoundError
        If ``ffprobe`` is not found on PATH.
    subprocess.CalledProcessError
        If ``ffprobe`` exits with a non-zero status.
    StopIteration
        If the file contains no video stream.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next(
        s for s in data["streams"] if s.get("codec_type") == "video"
    )

    stored_width = int(video_stream["width"])
    stored_height = int(video_stream["height"])

    fps_num, fps_den = str(video_stream["r_frame_rate"]).split("/")
    fps = float(fps_num) / float(fps_den)

    # Prefer Display Matrix side data (ffmpeg 5+); fall back to legacy tags.rotate.
    rotation_degrees = 0
    found_in_side_data = False
    for sd in video_stream.get("side_data_list") or []:
        if sd.get("side_data_type") == "Display Matrix":
            # Display Matrix rotation is the CCW angle of the matrix itself,
            # which is the negative of the CW rotation needed for display.
            raw = int(sd.get("rotation", 0))
            rotation_degrees = (-raw) % 360
            found_in_side_data = True
            break

    if not found_in_side_data:
        tags = video_stream.get("tags") or {}
        rotation_degrees = int(tags.get("rotate", 0))

    if rotation_degrees in (90, 270):
        display_width, display_height = stored_height, stored_width
    else:
        display_width, display_height = stored_width, stored_height

    nb_frames = video_stream.get("nb_frames")
    if nb_frames is not None and str(nb_frames) not in ("", "N/A"):
        frame_count = int(nb_frames)
    else:
        duration = float(video_stream.get("duration", "0"))
        frame_count = round(duration * fps)

    duration_sec = float(video_stream.get("duration", "0"))

    return VideoMetadata(
        path=path,
        width=display_width,
        height=display_height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        rotation_degrees=rotation_degrees,
    )


def iter_frames(path: Path) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield display-correct RGB frames decoded from a video file.

    Each frame is rotated to display orientation before being returned, so
    downstream code never needs to account for phone video rotation.

    Parameters
    ----------
    path : Path
        Path to the video file.

    Yields
    ------
    tuple[int, float, np.ndarray]
        ``(frame_idx, timestamp_sec, frame_rgb)`` where ``frame_rgb`` has
        shape ``(height, width, 3)`` in display orientation, dtype ``uint8``.
    """
    meta = probe_video(path)
    rotate_code = _ROTATE_MAP.get(meta.rotation_degrees)

    cap = cv2.VideoCapture(str(path))
    # Disable OpenCV's built-in auto-rotation; we apply it ourselves based on
    # ffprobe metadata so rotation is consistent across OpenCV versions.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            frame_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_idx, frame_idx / meta.fps, frame_rgb
            frame_idx += 1
    finally:
        cap.release()
