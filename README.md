# Kinema

Bouldering performance analytics from phone video. Feed it a climbing clip and
get back a pose-overlay video, keypoint time-series, move segmentation, and a
summary figure — all from a single command, no GPU required.

```
kinema run climb.mp4 \
    --out-dir results/ \
    # → results/overlay.mp4
    # → results/keypoints.parquet
    # → results/moves.csv
    # → results/summary.png
```
