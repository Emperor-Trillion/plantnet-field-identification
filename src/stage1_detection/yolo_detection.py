"""
Step 1.3: YOLO-based object detection.
Detects persons, plants, and other objects in each image.

CHANGED (Stage 1b fix):
- Loads excluded image list from 01_human_observations.csv before processing
- Skips images flagged contains_human=yes OR contains_flowers=no
- person_boxes are now written to JSON so flower_segmentation.py can use them
  to suppress overlapping ROIs
"""

import json
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from src.utils.image_utils import get_image_files


# ---------------------------------------------------------------------------
# Exclusion helpers
# ---------------------------------------------------------------------------

def load_excluded_images(observations_csv):
    """
    Read 01_human_observations.csv and return the set of image filenames
    that should be skipped entirely — images containing humans or no flowers.

    Returns empty set if the file does not exist (pipeline first run before
    observations are filled in — in that case YOLO runs on everything and
    the segmentation size/overlap filters act as the only safeguard).
    """
    obs_path = Path(observations_csv)
    if not obs_path.exists():
        print(f"   [WARN] Observations file not found: {obs_path}")
        print(f"   [WARN] Proceeding without image exclusion list.")
        return set()

    import pandas as pd
    df = pd.read_csv(obs_path)

    # Normalise column values to lowercase strings for robust comparison
    for col in ["contains_human", "contains_flowers"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    excluded = set()

    if "contains_human" in df.columns:
        human_images = df[df["contains_human"] == "yes"]["image_id"].tolist()
        excluded.update(human_images)
        if human_images:
            print(f"   Excluding {len(human_images)} images with humans: "
                  f"{human_images[:5]}{'...' if len(human_images) > 5 else ''}")

    if "contains_flowers" in df.columns:
        no_flower_images = df[df["contains_flowers"] == "no"]["image_id"].tolist()
        excluded.update(no_flower_images)
        if no_flower_images:
            print(f"   Excluding {len(no_flower_images)} images with no flowers.")

    print(f"   Total excluded images: {len(excluded)}")
    return excluded


# ---------------------------------------------------------------------------
# Per-image detection
# ---------------------------------------------------------------------------

def detect_objects_in_image(image_path, model, conf_threshold=0.25):
    """Run YOLO detection on a single image."""
    results = model.predict(
        source=str(image_path), conf=conf_threshold, save=False, verbose=False
    )

    detections = {
        "image_id": Path(image_path).name,
        "persons_detected": 0,
        "person_boxes": [],          # [x1, y1, x2, y2] in pixel coords
        "plants_detected": 0,
        "plant_boxes": [],
        "other_objects": [],
        "all_detections": [],
    }

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()   # [x1, y1, x2, y2]

            detection = {
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [round(c, 1) for c in bbox],
            }
            detections["all_detections"].append(detection)

            if cls_name == "person":
                detections["persons_detected"] += 1
                detections["person_boxes"].append([round(c, 1) for c in bbox])
            elif cls_name in ["potted plant", "vase"]:
                detections["plants_detected"] += 1
                detections["plant_boxes"].append([round(c, 1) for c in bbox])
            else:
                detections["other_objects"].append(detection)

    return detections


# ---------------------------------------------------------------------------
# Batch detection
# ---------------------------------------------------------------------------

def run_yolo_detection(
    image_dir,
    output_json,
    yolo_model_name="yolov8m.pt",
    conf_threshold=0.25,
    observations_csv=None,          # NEW: path to 01_human_observations.csv
):
    """
    Run YOLO detection on all images in directory.

    If observations_csv is provided, images flagged as containing humans
    or no flowers are skipped entirely — they will not appear in the output
    JSON and will therefore not produce any ROIs downstream.
    """
    model = YOLO(yolo_model_name)
    image_files = get_image_files(image_dir)

    # Load exclusion list from human observations
    excluded = set()
    if observations_csv:
        print("\n   Loading image exclusion list from observations...")
        excluded = load_excluded_images(observations_csv)

    all_detections = []
    skipped = 0

    for img_path in tqdm(image_files, desc="YOLO Detection"):
        if img_path.name in excluded:
            skipped += 1
            continue
        det = detect_objects_in_image(img_path, model, conf_threshold)
        all_detections.append(det)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(all_detections, f, indent=2)

    total_persons = sum(d["persons_detected"] for d in all_detections)
    total_plants = sum(d["plants_detected"] for d in all_detections)
    print(f"\nDetection Summary:")
    print(f"  Images in directory:      {len(image_files)}")
    print(f"  Skipped (excluded list):  {skipped}")
    print(f"  Images processed:         {len(all_detections)}")
    print(f"  Total persons detected:   {total_persons}")
    print(f"  Total plants detected:    {total_plants}")
    print(f"  Saved to: {output_json}")

    return all_detections


if __name__ == "__main__":
    from src.config import CONFIG

    csvs = CONFIG["paths"]["csvs"]
    run_yolo_detection(
        CONFIG["paths"]["raw_images"],
        f"{csvs}/02_yolo_detections.json",
        CONFIG["detection"]["yolo_model"],
        CONFIG["detection"]["yolo_confidence"],
        observations_csv=f"{csvs}/01_human_observations.csv",   # NEW
    )
