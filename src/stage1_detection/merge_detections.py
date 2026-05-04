"""
Step 1.5: Merge all detection results into master table.
Combines human observations + YOLO results + flower segmentation.
"""

import pandas as pd
import json
from pathlib import Path


def merge_all_detections(observations_csv, yolo_json, rois_csv, output_csv):
    """Merge all Stage 1 outputs into a single master table"""

    # Load human observations
    df_human = pd.read_csv(observations_csv)

    # Parse YOLO results
    with open(yolo_json) as f:
        detections = json.load(f)

    yolo_summary = []
    for det in detections:
        yolo_summary.append(
            {
                "image_id": det["image_id"],
                "yolo_persons": det["persons_detected"],
                "yolo_plants": det["plants_detected"],
                "yolo_other_count": len(det["other_objects"]),
                "yolo_other_classes": ", ".join(
                    [d["class"] for d in det["other_objects"]]
                ),
            }
        )
    df_yolo = pd.DataFrame(yolo_summary)

    # Count ROIs per image
    df_rois = pd.read_csv(rois_csv)
    roi_counts = (
        df_rois.groupby("source_image").size().reset_index(name="num_flower_rois")
    )
    roi_counts.rename(columns={"source_image": "image_id"}, inplace=True)

    # Merge
    df_master = df_human.merge(df_yolo, on="image_id", how="left")
    df_master = df_master.merge(roi_counts, on="image_id", how="left")
    df_master["num_flower_rois"] = df_master["num_flower_rois"].fillna(0).astype(int)

    # Auto-categorize scenes
    def categorize_image(row):
        categories = []
        if row["num_flower_rois"] > 0:
            categories.append("flowers")
        if row["yolo_persons"] > 0:
            categories.append("humans")
        if row["num_flower_rois"] > 1:
            categories.append("multi_flower")
        if row["num_flower_rois"] == 0:
            categories.append("no_flower_detected")
        return "|".join(categories) if categories else "unknown"

    df_master["scene_category"] = df_master.apply(categorize_image, axis=1)

    df_master.to_csv(output_csv, index=False)

    print(f"\nMaster Table Summary:")
    print(f"  Total images: {len(df_master)}")
    print(f"\n  Scene Categories:")
    print(df_master["scene_category"].value_counts().to_string())

    return df_master


if __name__ == "__main__":
    from src.config import CONFIG

    csvs = CONFIG["paths"]["csvs"]
    merge_all_detections(
        f"{csvs}/01_human_observations.csv",
        f"{csvs}/02_yolo_detections.json",
        f"{csvs}/03_flower_rois.csv",
        f"{csvs}/04_master_image_table.csv",
    )
