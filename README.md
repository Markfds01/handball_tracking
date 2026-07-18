

# Handball Player Tracking → Tactical Map

Tracking handball players from broadcast video and projecting them onto a
top-down tactical map, built around a **fine-tuned RF-DETR** detector.

https://github.com/user-attachments/assets/b6b59e4b-94b5-4458-bfec-021f93de56fd

> Demo: Sporting CP vs Veszprém HC. Top — RF-DETR detections + ByteTrack IDs,
> colored by team. Bottom — players projected onto the court via homography.
> ([animated GIF fallback](assets/demo.gif))

## Pipeline

1. **Player detection** — RF-DETR Medium, fine-tuned on a single `player` class.
2. **Tracking** — ByteTrack assigns persistent IDs across frames.
3. **Team classification** — unsupervised: SigLIP embeddings → UMAP → KMeans
   split players into two teams by jersey; display colors are auto-aligned to
   the real kit colors.
4. **Court keypoints** — a YOLOv8-pose model detects 28 court landmarks.
5. **Homography** — landmarks → a top-down court; player feet are projected onto
   the tactical map.

## Setup

```bash
pip install -r requirements.txt
# place the two checkpoints in weights/ (see weights/README.md)
```

## Run

```bash
python track_and_map.py --source your_clip.mp4 --output demo.mp4
python track_and_map.py --source your_clip.mp4 --no-teams   # single color
```

## Training

Both models are trained from your own datasets (replace the `your/data`
placeholders with your dataset paths):

```bash
python train/train_rfdetr.py      # player detector  -> runs/rfdetr_players/
python train/train_keypoints.py   # court keypoints   -> handball_runs/pose_model/
```

## Layout

```
track_and_map.py        inference: detect → track → teams → homography → map
train/train_rfdetr.py   RF-DETR player detector training
train/train_keypoints.py YOLOv8-pose court-keypoint training
sports/handball/        court geometry, tactical-map drawing, team classifier
assets/demo.mp4         example output
weights/                trained checkpoints (not tracked; see README there)
```

## Notes

- The court-keypoint model must be trained on footage similar to your target
  broadcast. If it only detects center-court points (clustered on the midline),
  the homography degenerates — add representative frames to its training set.

## Acknowledgements

Heavily inspired by **[roboflow/sports](https://github.com/roboflow/sports)** by
Piotr Skalski (Roboflow) — the tactical-map / homography approach and the
SigLIP + UMAP + KMeans team classifier follow that project's design, adapted
here for handball.

## Contact

Trained weights are **available on request** — I'm happy to share them.
Open an issue on this repo or email <markfds1228@gmail.com>.
