"""
Handball player tracking + tactical map (RF-DETR + ByteTrack + homography).

Pipeline:
  1. RF-DETR (fine-tuned) detects players.
  2. ByteTrack assigns persistent IDs across frames.
  3. An unsupervised TeamClassifier (SigLIP + UMAP + KMeans) splits players
     into two teams from their jersey appearance.
  4. A YOLOv8-pose model detects court keypoints -> homography to a top-down court.
  5. Player feet are projected onto a tactical map, colored by team.
  6. Output = the annotated broadcast frame stacked on top of the tactical map.

Usage:
  python track_and_map.py --source spo-vez.mp4 --output demo.mp4
  python track_and_map.py --source spo-vez.mp4 --no-teams   # single color
"""
import argparse
import cv2
import numpy as np
import supervision as sv
from PIL import Image
from tqdm import tqdm
from rfdetr import RFDETRMedium
from ultralytics import YOLO

from sports.handball.keypoints_dict import HandballCourt
from sports.handball.draw_court import create_court_image
from sports.handball.teamclassifier import TeamClassifier

PLAYER_MODEL = "weights/rfdetr_players.pth"
KEYPOINT_MODEL = "weights/court_keypoints.pt"
PLAYER_CONFIDENCE = 0.45
KEYPOINT_CONFIDENCE = 0.6

# Team colors (BGR for OpenCV, hex for supervision annotators)
TEAM_HEX = ["#FF3131", "#00A000"]          # team 0 red, team 1 green
TEAM_BGR = [(49, 49, 255), (0, 160, 0)]
FIT_STRIDE = 5        # sample every Nth frame to fit the team clusters
RECLASSIFY_STRIDE = 15  # refresh a tracker's team every N frames


def collect_crops(video, player_model, device):
    """Sample player crops across the video to fit the TeamClassifier."""
    crops = []
    for frame in tqdm(sv.get_video_frames_generator(video, stride=FIT_STRIDE),
                      desc="Collecting crops"):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        det = player_model.predict(img_pil, threshold=0.6)
        for box in sv.scale_boxes(det.xyxy, factor=0.5):
            crop = sv.crop_image(frame, box)
            if crop.size:
                crops.append(crop)
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="spo-vez.mp4")
    ap.add_argument("--output", default="demo_tracking.mp4")
    ap.add_argument("--no-teams", action="store_true", help="disable team classification")
    args = ap.parse_args()
    use_teams = not args.no_teams

    print("Loading models...")
    player_model = RFDETRMedium(num_classes=1, pretrain_weights=PLAYER_MODEL)
    keypoint_model = YOLO(KEYPOINT_MODEL)
    tracker = sv.ByteTrack(frame_rate=25)

    team_classifier = None
    # team_id -> display slot (0=red, 1=green). Aligned to real jersey colors below.
    slot_of_team = {0: 0, 1: 1}
    if use_teams:
        print("Fitting team classifier...")
        crops = collect_crops(args.source, player_model, "cuda")
        team_classifier = TeamClassifier(device="cuda")
        team_classifier.fit(crops)
        # Align cluster labels to jersey colors: whichever cluster's crops are
        # "redder" on average gets the red display slot.
        labels = team_classifier.predict(crops)
        redness = {}
        for t in (0, 1):
            px = [c[c.shape[0] // 4:3 * c.shape[0] // 4,
                    c.shape[1] // 4:3 * c.shape[1] // 4].reshape(-1, 3).mean(0)
                  for c, l in zip(crops, labels) if l == t and c.size]
            m = np.mean(px, axis=0) if px else np.zeros(3)  # BGR
            redness[t] = m[2] - m[1]                        # R - G
        red_team = max(redness, key=redness.get)
        slot_of_team = {red_team: 0, 1 - red_team: 1}
        print(f"Team color alignment: cluster {red_team} -> red, "
              f"cluster {1 - red_team} -> green")

    palette = sv.ColorPalette.from_hex(TEAM_HEX if use_teams else ["#FF3131"])
    box_annotator = sv.BoxAnnotator(color=palette, thickness=2,
                                    color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color=palette, text_color=sv.Color.WHITE,
                                        text_scale=0.5, text_thickness=1,
                                        color_lookup=sv.ColorLookup.CLASS)

    court = HandballCourt()
    world_ref = np.full((28, 2), np.nan, dtype=np.float32)
    for k, c in court.keypoints.items():
        world_ref[k] = c

    base_court, ppm, margin = create_court_image(height=1080)
    court_h, court_w, _ = base_court.shape

    info = sv.VideoInfo.from_video_path(args.source)
    W, H = info.width, info.height
    scale = W / court_w
    map_h = int(court_h * scale)
    out_info = sv.VideoInfo(width=W, height=H + map_h, fps=info.fps)

    team_of = {}  # tracker_id -> team_id (cache)
    idx = 0
    frame_gen = sv.get_video_frames_generator(args.source)
    print(f"Processing {info.total_frames} frames -> {args.output}")

    with sv.VideoSink(args.output, out_info) as sink:
        for frame in tqdm(frame_gen, total=info.total_frames):
            # 1. DETECT (RF-DETR)  2. TRACK (ByteTrack)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = player_model.predict(img_pil, threshold=PLAYER_CONFIDENCE)
            detections = tracker.update_with_detections(detections)

            # 3. TEAM CLASSIFICATION (cached per tracker id, refreshed periodically)
            if use_teams and len(detections) > 0:
                boxes = sv.scale_boxes(detections.xyxy, factor=0.5)
                todo_i, todo_crops = [], []
                for i, tid in enumerate(detections.tracker_id):
                    if tid not in team_of or idx % RECLASSIFY_STRIDE == 0:
                        crop = sv.crop_image(frame, boxes[i])
                        if crop.size:
                            todo_i.append(i)
                            todo_crops.append(crop)
                if todo_crops:
                    preds = team_classifier.predict(todo_crops)
                    for i, t in zip(todo_i, preds):
                        team_of[detections.tracker_id[i]] = int(t)
                detections.class_id = np.array(
                    [slot_of_team[team_of.get(tid, 0)] for tid in detections.tracker_id])
            elif len(detections) > 0:
                detections.class_id = np.zeros(len(detections), dtype=int)

            annotated = frame.copy()
            annotated = box_annotator.annotate(annotated, detections)
            labels = [f"#{tid}" for tid in detections.tracker_id]
            annotated = label_annotator.annotate(annotated, detections, labels)

            # 4. COURT KEYPOINTS -> HOMOGRAPHY
            r = keypoint_model(cv2.resize(frame, (640, 640)), conf=0.5, verbose=False)[0]
            court_img = base_court.copy()
            if r.keypoints.xy.numel() > 0:
                xyn = r.keypoints.xyn[0].cpu().numpy()
                conf = r.keypoints.conf[0].cpu().numpy()
                pred_xy = xyn * np.array([W, H])
                mask = (conf > KEYPOINT_CONFIDENCE) & (~np.isnan(world_ref).any(axis=1))
                dst, src = pred_xy[mask], world_ref[mask]
                if len(dst) >= 4:
                    Hmat, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
                    if Hmat is not None and len(detections) > 0:
                        feet = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        pts = cv2.perspectiveTransform(
                            feet.reshape(-1, 1, 2).astype(np.float32), Hmat).reshape(-1, 2)
                        # 5. DRAW ON TACTICAL MAP (colored by team)
                        for (mx, my), tid, cid in zip(pts, detections.tracker_id,
                                                      detections.class_id):
                            if np.isnan(mx) or np.isnan(my):
                                continue
                            px = int((mx + margin) * ppm)
                            py = int((my + margin) * ppm)
                            if 0 <= px < court_w and 0 <= py < court_h:
                                color = TEAM_BGR[int(cid)] if use_teams else TEAM_BGR[0]
                                cv2.circle(court_img, (px, py), 14, color, -1)
                                cv2.circle(court_img, (px, py), 14, (255, 255, 255), 2)
                                cv2.putText(court_img, str(tid), (px - 8, py + 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 6. STACK
            map_resized = cv2.resize(court_img, (W, map_h))
            sink.write_frame(np.vstack([annotated, map_resized]))
            idx += 1

    print("Done:", args.output)


if __name__ == "__main__":
    main()
