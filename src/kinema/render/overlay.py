"""Skeleton overlay rendering: keypoints → overlay.mp4."""

import logging
import math
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from kinema.io.com import read_com
from kinema.io.keypoints import read_keypoints
from kinema.io.video import iter_frames, probe_video

logger = logging.getLogger(__name__)

# BGR colors
_SKELETON_COLOR: tuple[int, int, int] = (255, 255, 0)  # cyan
_COM_COLOR: tuple[int, int, int] = (255, 0, 255)       # magenta
_LANDMARK_RADIUS: int = 4
_SKELETON_THICKNESS: int = 2
_COM_RADIUS: int = 8

# Standard BlazePose 33-landmark connections (same as mediapipe.solutions.pose.POSE_CONNECTIONS).
POSE_CONNECTIONS: list[tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Shoulders
    (11, 12),
    # Left arm + hand
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm + hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Left leg + foot
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Right leg + foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def render_overlay(
    video_path: Path,
    keypoints_path: Path,
    output_path: Path,
    *,
    com_path: Path | None = None,
    visibility_threshold: float = 0.5,
) -> None:
    """Draw pose skeleton on each video frame and write output video.

    Parameters
    ----------
    video_path : Path
        Source video file.
    keypoints_path : Path
        Keypoints Parquet file produced by ``kinema.pose.extract``.
    output_path : Path
        Destination overlay video (mp4v codec). Overwritten if it exists.
    com_path : Path | None
        Optional COM trajectory Parquet file. When provided, a magenta marker
        and 1-second fading trail are drawn on each frame.
    visibility_threshold : float
        Minimum landmark visibility score in [0, 1] to render a landmark.
    """
    meta = probe_video(video_path)
    trail_maxlen = max(1, round(meta.fps))

    kp_df = read_keypoints(keypoints_path)
    # frame_idx → {landmark_id: (x_norm, y_norm, visibility)}
    kp_by_frame: dict[int, dict[int, tuple[float, float, float]]] = {}
    for row in kp_df.itertuples(index=False):
        r: Any = row
        fidx = int(r.frame_idx)
        lid = int(r.landmark_id)
        kp_by_frame.setdefault(fidx, {})[lid] = (float(r.x), float(r.y), float(r.visibility))

    com_by_frame: dict[int, tuple[float, float]] = {}
    if com_path is not None:
        com_df = read_com(com_path)
        for row in com_df.itertuples(index=False):
            r2: Any = row
            com_by_frame[int(r2.frame_idx)] = (float(r2.com_x), float(r2.com_y))

    # mp4v: widely playable without extra OpenCV build flags.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, meta.fps, (meta.width, meta.height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_path}")

    com_trail: deque[tuple[int, int]] = deque(maxlen=trail_maxlen)

    try:
        for frame_idx, _ts, frame_rgb in iter_frames(video_path):
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]

            lm = kp_by_frame.get(frame_idx, {})

            # Draw skeleton edges before landmarks so circles appear on top.
            for a, b in POSE_CONNECTIONS:
                if a in lm and b in lm:
                    ax, ay, av = lm[a]
                    bx, by, bv = lm[b]
                    if av >= visibility_threshold and bv >= visibility_threshold:
                        cv2.line(
                            bgr,
                            (int(ax * w), int(ay * h)),
                            (int(bx * w), int(by * h)),
                            _SKELETON_COLOR,
                            _SKELETON_THICKNESS,
                        )

            for _lid, (x, y, vis) in lm.items():
                if vis >= visibility_threshold:
                    cv2.circle(bgr, (int(x * w), int(y * h)), _LANDMARK_RADIUS, _SKELETON_COLOR, -1)

            if com_by_frame:
                com_xy = com_by_frame.get(frame_idx)
                if com_xy is not None:
                    cx, cy = com_xy
                    if not (math.isnan(cx) or math.isnan(cy)):
                        com_trail.append((int(cx * w), int(cy * h)))

                trail_list = list(com_trail)
                n = len(trail_list)
                for age, pt in enumerate(reversed(trail_list)):
                    # age 0 = newest; scale radius by recency fraction
                    frac = 1.0 - age / max(1, n)
                    r = max(1, round(_COM_RADIUS * frac))
                    cv2.circle(bgr, pt, r, _COM_COLOR, -1)

            writer.write(bgr)
            logger.debug("Rendered overlay frame %d", frame_idx)
    finally:
        writer.release()

    logger.info("Overlay written → %s", output_path)
