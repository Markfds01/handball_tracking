"""
Train the player detector (RF-DETR Medium, single "player" class).

RF-DETR is fine-tuned on a handball player-detection dataset exported in
COCO format. The resulting checkpoint is used by track_and_map.py for
detection + tracking.

Replace the dataset path with your own COCO-format player dataset.
"""
from rfdetr import RFDETRMedium

# Path to your COCO-format player-detection dataset root
# (must contain train/ valid/ test/ with _annotations.coco.json).
# e.g. "your/data/handball_players"
DATASET_DIR = "your/data/handball_players"

if __name__ == "__main__":
    model = RFDETRMedium(num_classes=1, device="cuda")
    model.train(
        dataset_dir=DATASET_DIR,
        epochs=50,
        batch_size=4,
        grad_accum_steps=2,
        lr=1e-4,
        output_dir="runs/rfdetr_players",
    )
    # Best weights -> runs/rfdetr_players/checkpoint_best_total.pth
