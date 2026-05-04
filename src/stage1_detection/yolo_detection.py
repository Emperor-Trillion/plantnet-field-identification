"""
Step 1.3: YOLO-based object detection.
Detects persons, plants, and other objects in each image.
"""

import json
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from src.utils.image_utils import get_image_files


def detect_objects_in_image(image_path, model, conf_threshold=0.25):
    """Run YOLO detection on a single image"""
    results = model.predict(
        source=str(image_path), conf=conf_threshold, save=False, verbose=False
    )

    detections = {
        "image_id": Path(image_path).name,
        "persons_detected": 0,
        "person_boxes": [],
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
            bbox = box.xyxy[0].tolist()

            detection = {
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [round(c, 1) for c in bbox],
            }
            detections["all_detections"].append(detection)

            if cls_name == "person":
                detections["persons_detected"] += 1
                detections["person_boxes"].append(bbox)
            elif cls_name in ["potted plant", "vase"]:
                detections["plants_detected"] += 1
                detections["plant_boxes"].append(bbox)
            else:
                detections["other_objects"].append(detection)

    return detections


def run_yolo_detection(
    image_dir, output_json, yolo_model_name="yolov8m.pt", conf_threshold=0.25
):
    """Run YOLO detection on all images in directory"""
    model = YOLO(yolo_model_name)
    image_files = get_image_files(image_dir)

    all_detections = []
    for img_path in tqdm(image_files, desc="YOLO Detection"):
        det = detect_objects_in_image(img_path, model, conf_threshold)
        all_detections.append(det)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(all_detections, f, indent=2)

    # Print summary
    total_persons = sum(d["persons_detected"] for d in all_detections)
    total_plants = sum(d["plants_detected"] for d in all_detections)
    print(f"\nDetection Summary:")
    print(f"  Images processed: {len(all_detections)}")
    print(f"  Total persons detected: {total_persons}")
    print(f"  Total plants detected: {total_plants}")
    print(f"  Saved to: {output_json}")

    return all_detections


if __name__ == "__main__":
    from src.config import CONFIG

    run_yolo_detection(
        CONFIG["paths"]["raw_images"],
        f"{CONFIG['paths']['csvs']}/02_yolo_detections.json",
        CONFIG["detection"]["yolo_model"],
        CONFIG["detection"]["yolo_confidence"],
    )
