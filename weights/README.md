# Weights

The pipeline needs two trained checkpoints (not tracked in git — each > 100 MB):

| File | Model | Produced by |
|------|-------|-------------|
| `weights/rfdetr_players.pth` | RF-DETR Medium player detector (1 class) | `train/train_rfdetr.py` |
| `weights/court_keypoints.pt` | YOLOv8-pose court keypoints (28 pts) | `train/train_keypoints.py` |

Place both files in this folder before running `track_and_map.py`.

Either train them yourself with the scripts in `train/`, or host the
checkpoints (GitHub Releases / cloud storage) and download them here.
