"""MediaPipe Pose extraction: video → keypoints Parquet."""

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from kinema.io.keypoints import LANDMARK_NAMES, write_keypoints
from kinema.io.video import iter_frames, probe_video

logger = logging.getLogger(__name__)

_N_LANDMARKS = len(LANDMARK_NAMES)
_CACHE_DIR = Path.home() / ".cache" / "kinema" / "models"

_MODEL_URLS: dict[int, str] = {
    0: (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    1: (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    2: (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
}
_MODEL_FILENAMES: dict[int, str] = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


@dataclass(frozen=True)
class ExtractionStats:
    """Summary statistics from a pose extraction run.

    Parameters
    ----------
    frame_count : int
        Total frames processed.
    frames_with_pose : int
        Frames where at least one pose was detected.
    mean_visibility : float
        Mean visibility across all landmarks and frames (0.0 if no pose detected).
    wall_time_sec : float
        Wall-clock time for the extraction in seconds.
    """

    frame_count: int
    frames_with_pose: int
    mean_visibility: float
    wall_time_sec: float


def _get_model_path(model_complexity: int) -> Path:
    """Return path to the pose landmarker .task model, downloading if absent.

    Parameters
    ----------
    model_complexity : int
        0 = lite, 1 = full, 2 = heavy.

    Returns
    -------
    Path
        Local path to the model file.

    Raises
    ------
    ValueError
        If ``model_complexity`` is not 0, 1, or 2.
    """
    if model_complexity not in _MODEL_URLS:
        raise ValueError(f"model_complexity must be 0, 1, or 2; got {model_complexity}")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _CACHE_DIR / _MODEL_FILENAMES[model_complexity]

    if not model_path.exists():
        url = _MODEL_URLS[model_complexity]
        logger.info("Downloading pose model to %s", model_path)
        urllib.request.urlretrieve(url, str(model_path))
        logger.info("Model download complete: %s", model_path.name)

    return model_path


def extract_pose(
    video_path: Path,
    output_path: Path,
    *,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> ExtractionStats:
    """Extract pose keypoints from a video and write to Parquet.

    Processes the video frame-by-frame using MediaPipe Pose (Tasks API, VIDEO
    mode). Frames where no pose is detected produce 33 rows with NaN x/y/z and
    visibility=0.0. Writes the resulting DataFrame via
    :func:`kinema.io.keypoints.write_keypoints`.

    Parameters
    ----------
    video_path : Path
        Input video file. Rotation metadata is handled by ``iter_frames``.
    output_path : Path
        Destination Parquet file (overwritten if present).
    model_complexity : int
        MediaPipe model variant: 0=lite, 1=full, 2=heavy.
    min_detection_confidence : float
        Minimum confidence for initial pose detection.
    min_tracking_confidence : float
        Minimum confidence for pose tracking across frames.

    Returns
    -------
    ExtractionStats
        Frame counts, mean visibility, and wall-clock time.
    """
    model_path = _get_model_path(model_complexity)
    meta = probe_video(video_path)

    logger.info(
        "Extracting pose from %s: %d frames at %.1f fps",
        video_path.name,
        meta.frame_count,
        meta.fps,
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frame_indices: list[int] = []
    timestamps_ms: list[int] = []
    landmark_ids: list[int] = []
    landmark_names: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    visibilities: list[float] = []

    frames_with_pose = 0
    t_start = time.perf_counter()

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_idx, timestamp_sec, frame_rgb in tqdm(
            iter_frames(video_path),
            total=meta.frame_count,
            desc=video_path.name,
            unit="frame",
        ):
            ts_ms = round(timestamp_sec * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            pose_detected = bool(result.pose_landmarks)
            if pose_detected:
                frames_with_pose += 1
                landmarks = result.pose_landmarks[0]
            else:
                landmarks = None

            for lid, name in enumerate(LANDMARK_NAMES):
                frame_indices.append(frame_idx)
                timestamps_ms.append(ts_ms)
                landmark_ids.append(lid)
                landmark_names.append(name)

                if landmarks is not None:
                    lm = landmarks[lid]
                    xs.append(float(lm.x))
                    ys.append(float(lm.y))
                    zs.append(float(lm.z))
                    vis = lm.visibility
                    visibilities.append(float(vis) if vis is not None else 0.0)
                else:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                    zs.append(float("nan"))
                    visibilities.append(0.0)

    wall_time_sec = time.perf_counter() - t_start

    df = pd.DataFrame(
        {
            "frame_idx": np.array(frame_indices, dtype=np.int32),
            "timestamp_ms": np.array(timestamps_ms, dtype=np.int64),
            "landmark_id": np.array(landmark_ids, dtype=np.int8),
            "landmark_name": pd.Categorical(
                landmark_names, categories=list(LANDMARK_NAMES)
            ),
            "x": np.array(xs, dtype=np.float32),
            "y": np.array(ys, dtype=np.float32),
            "z": np.array(zs, dtype=np.float32),
            "visibility": np.array(visibilities, dtype=np.float32),
        }
    )

    write_keypoints(df, output_path)

    frame_count = len(frame_indices) // _N_LANDMARKS
    mean_visibility = float(np.mean(visibilities)) if visibilities else 0.0

    logger.info(
        "Extraction complete: %d/%d frames with pose, %.3f mean visibility, %.1fs",
        frames_with_pose,
        frame_count,
        mean_visibility,
        wall_time_sec,
    )

    return ExtractionStats(
        frame_count=frame_count,
        frames_with_pose=frames_with_pose,
        mean_visibility=mean_visibility,
        wall_time_sec=wall_time_sec,
    )
