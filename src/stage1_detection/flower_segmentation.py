"""
Step 1.4: Color-based flower segmentation.
Isolates flower regions and saves cropped ROIs.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils.image_utils import get_image_files


def segment_flower_regions(
    image_path,
    sat_threshold=80,
    green_hue_range=(35, 85),
    min_area_fraction=0.01,
    padding=20,
):
    """
    Segment likely flower regions using color analysis.
    Flowers = high saturation AND not green.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return [], None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    sat_mask = s > sat_threshold
    green_mask = (h > green_hue_range[0]) & (h < green_hue_range[1])
    non_green_mask = ~green_mask
    flower_mask = (sat_mask & non_green_mask).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    flower_mask = cv2.morphologyEx(flower_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    flower_mask = cv2.morphologyEx(flower_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        flower_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = img.shape[0] * img.shape[1] * min_area_fraction
    flower_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h_box = cv2.boundingRect(contour)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h_box + padding)

            flower_regions.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "area": int(area),
                    "center": [x + w // 2, y + h_box // 2],
                }
            )

    return flower_regions, flower_mask


def segment_all_images(image_dir, roi_output_dir, csv_output_path, config=None):
    """Process all images: segment flowers and save ROI crops"""
    image_files = get_image_files(image_dir)
    roi_dir = Path(roi_output_dir)
    roi_dir.mkdir(parents=True, exist_ok=True)

    sat_thresh = config["detection"]["flower_saturation_threshold"] if config else 80
    green_range = (
        tuple(config["detection"]["flower_green_hue_range"]) if config else (35, 85)
    )
    min_area = config["detection"]["min_roi_area_fraction"] if config else 0.01
    pad = config["detection"]["roi_padding"] if config else 20

    roi_records = []
    roi_counter = 0

    for img_path in tqdm(image_files, desc="Segmenting flowers"):
        regions, mask = segment_flower_regions(
            img_path, sat_thresh, green_range, min_area, pad
        )
        img = cv2.imread(str(img_path))

        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region["bbox"]
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            roi_id = f"roi_{roi_counter:04d}"
            crop_path = roi_dir / f"{roi_id}.jpg"
            cv2.imwrite(str(crop_path), crop)

            roi_records.append(
                {
                    "roi_id": roi_id,
                    "source_image": img_path.name,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "area": region["area"],
                    "center_x": region["center"][0],
                    "center_y": region["center"][1],
                    "crop_path": str(crop_path),
                }
            )
            roi_counter += 1

    df_rois = pd.DataFrame(roi_records)
    df_rois.to_csv(csv_output_path, index=False)

    print(f"\nSegmentation Summary:")
    print(f"  Images processed: {len(image_files)}")
    print(f"  Total ROIs extracted: {len(df_rois)}")
    print(f"  Avg ROIs per image: {len(df_rois)/len(image_files):.1f}")
    print(f"  Saved crops to: {roi_output_dir}")
    print(f"  Saved CSV to: {csv_output_path}")

    return df_rois


if __name__ == "__main__":
    from src.config import CONFIG

    segment_all_images(
        CONFIG["paths"]["raw_images"],
        CONFIG["paths"]["cropped_rois"],
        f"{CONFIG['paths']['csvs']}/03_flower_rois.csv",
        CONFIG,
    )
