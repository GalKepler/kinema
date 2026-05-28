# Kinema

Bouldering performance analytics from phone video. Feed it a climbing clip and
get back a pose-overlay video, keypoint time-series, move segmentation, and a
summary figure — all from a single command, no GPU required.

## System dependencies

`ffmpeg` and `ffprobe` (≥ 5.0) must be on `PATH`. They are used to read
rotation metadata that OpenCV ignores by default and to generate test videos.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

## Usage

```
kinema run climb.mp4 \
    --out-dir results/ \
    # → results/overlay.mp4
    # → results/keypoints.parquet
    # → results/moves.csv
    # → results/summary.png
```
