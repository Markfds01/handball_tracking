"""
Train the court-keypoint model (YOLOv8-pose, 28 court keypoints).

The model detects handball-court landmarks (goal posts, 6m/9m arcs, corners,
center circle) which are used to compute the homography for the tactical map.

Replace the dataset path with your own YOLO-pose keypoint dataset
(exported e.g. from Roboflow / CVAT in the YOLOv8 pose format).
"""
from ultralytics import YOLO

# Path to your dataset's data.yaml (YOLOv8 pose format).
# e.g. "your/data/handball_keypoints/data.yaml"
DATA_YAML = "your/data/handball_keypoints/data.yaml"

if __name__ == "__main__":
    model = YOLO("yolov8x-pose.pt")  # pretrained pose backbone
    model.train(
        data=DATA_YAML,
        task="pose",
        project="handball_runs",
        name="pose_model",
        epochs=100,
        imgsz=640,
        batch=16,
        mosaic=0.0,
        plots=True,
        device=0,
    )
    # Best weights -> handball_runs/pose_model/weights/best.pt
