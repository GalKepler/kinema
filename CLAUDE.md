# Kinema

Bouldering performance analytics from phone video. Phone video in →
pose overlay + movement metrics out.

The name is from "kinematics" — the project is fundamentally about
measuring how climbers move.

## Owner context
PhD candidate / data scientist transitioning to ML engineering. Strong Python,
ML pipelines, scientific computing. Weaker on computer vision, frontend,
consumer product. This project is partly a learning vehicle for those gaps.

## Long-term direction (not current scope)
The end target is a mobile app (iOS/Android, likely React Native, undecided).
This does not change the two-week MVP. It only means:
- Keep coordinates normalized, not pixel-based.
- Handle phone video rotation correctly in io/video.py.
- Don't introduce desktop-only heavy dependencies in the core pipeline.
Do not propose mobile frontend work, on-device inference, or model conversion
until I explicitly open that phase.

## Current scope (two-week MVP)
- Pose extraction pipeline: video → keypoints Parquet
- Skeleton overlay video
- Center-of-mass trajectory
- Move segmentation from velocity (smoothed)
- Smoothness metric (normalized jerk on COM)
- Single CLI: video in → overlay.mp4 + keypoints.parquet + moves.csv + summary.png

## Explicitly out of scope until week 3+
Hold detection, frontend (web or mobile), backend, database, user accounts,
deployment, grade prediction, embedding training, on-device inference,
model quantization or conversion, multi-user support, auth, cloud anything.
If a suggestion touches these, flag it and stop.

## Stack (decided, do not relitigate)
- Python 3.11, uv for dependency management
- MediaPipe Pose, CPU only
- OpenCV for video I/O
- Parquet (pyarrow) for keypoint time-series
- Click for CLI, Pydantic for config
- pytest, ruff, mypy strict
- matplotlib for the summary page

## Code conventions
- src/ layout. All code under src/kinema/.
- Type hints on every function signature. mypy strict.
- All functions have numpy-style docstrings.
- Parquet schema is defined ONCE in kinema.io.keypoints. Import from there.
- No print(); use logging. CLI configures logging in cli.py.
- pathlib.Path for all file paths, never strings.
- No notebooks committed with outputs.
- Tests in tests/, mirror the src/ layout.
- Line length 100, ruff enforced.

## Pipeline contract
Each stage reads from disk and writes to disk. Stages are independently runnable.

video.mp4
  → kinema.pose.extract        → keypoints.parquet
  → kinema.kinematics.com      → com trajectory (sidecar parquet)
  → kinema.segmentation.moves  → moves.csv
  → kinema.render.overlay      → overlay.mp4
  → kinema.render.summary      → summary.png

Do not fuse stages. Do not pass numpy arrays across module boundaries
when a Parquet file is the contract.

## Working style
- Propose a plan before writing code for any task that touches more than one file.
- When uncertain about a library API, run a tiny script to verify rather than guessing.
- After making changes, run: ruff check, mypy, pytest. Report results.
- If a test fails, fix it or explain why it's expected. Don't skip or xfail silently.
- Commit messages: imperative mood, lowercase, scope prefix.
  Examples: "io: add keypoints parquet schema", "pose: handle missing landmarks"
- Don't commit on my behalf unless I ask. Stage changes; let me review.
- Use detailed Jupyter Notebooks to showcase the analyses progress when suitable.

## What to push back on
- Scope creep into out-of-scope items above.
- Adding dependencies not already listed without justification.
- "Helpful" features I didn't ask for (web UI, Docker, CI, fancy logging frameworks).
- Premature abstraction. Concrete first, abstract on the third repetition.