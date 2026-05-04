"""
Compare model performance: full images vs cropped ROIs.
This gives meaningful metrics without external ground truth.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from src.utils.model_utils import load_plantnet_model, load_species_mapping
from src.utils.image_utils import (
    load_image_pil,
    get_preprocess_transform,
    get_image_files,
)


def compare_full_vs_roi(
    roi_csv,
    final_labels_csv,
    raw_image_dir,
    weights_path,
    class_idx_path,
    species_id_to_name_path,
    output_dir,
    device="cpu",
):
    """
    Use ROI predictions as reference labels.
    Then classify FULL images and measure how often
    the full-image prediction matches the ROI prediction.

    This measures: "Does detection + cropping improve classification?"
    """
    output_dir = Path(output_dir)

    df_rois = pd.read_csv(roi_csv)
    df_labels = pd.read_csv(final_labels_csv)

    model = load_plantnet_model(weights_path, class_idx_path, device=device)
    species_map = load_species_mapping(class_idx_path, species_id_to_name_path)
    preprocess = get_preprocess_transform()
    name_to_idx = {v: k for k, v in species_map.items()}

    # Only use high-confidence ROI labels as reference
    df_confident = df_labels[df_labels["label_confidence"] == "high"]

    # Get one ROI per source image (the most confident one)
    # Group by source image, take the first confident label
    image_reference = {}
    for _, row in df_confident.iterrows():
        img = row["source_image"]
        if img not in image_reference:
            species = row["final_species"]
            if species in name_to_idx:
                image_reference[img] = {
                    "species": species,
                    "genus": row["final_genus"],
                    "class_idx": name_to_idx[species],
                }

    print(f"Images with confident ROI reference labels: {len(image_reference)}")

    # Now classify full images and compare
    roi_top1_correct = 0
    roi_top5_correct = 0
    full_top1_correct = 0
    full_top5_correct = 0

    roi_confidences = []
    full_confidences = []

    per_image_results = []
    total = 0

    for img_name, ref in tqdm(image_reference.items(), desc="Comparing full vs ROI"):
        true_idx = ref["class_idx"]

        # Classify FULL image
        full_path = Path(raw_image_dir) / img_name
        if not full_path.exists():
            continue

        try:
            img = load_image_pil(full_path)
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            full_top1 = np.argmax(probs)
            full_top5 = np.argsort(probs)[-5:]
            full_conf = probs[full_top1]

            full_top1_match = int(full_top1 == true_idx)
            full_top5_match = int(true_idx in full_top5)

            full_top1_correct += full_top1_match
            full_top5_correct += full_top5_match
            full_confidences.append(full_conf)

        except Exception as e:
            continue

        # Classify ROI crop (find corresponding ROI)
        roi_rows = df_rois[df_rois["source_image"] == img_name]
        if len(roi_rows) == 0:
            continue

        try:
            roi_path = roi_rows.iloc[0]["crop_path"]
            img = load_image_pil(roi_path)
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            roi_top1 = np.argmax(probs)
            roi_top5 = np.argsort(probs)[-5:]
            roi_conf = probs[roi_top1]

            roi_top1_match = int(roi_top1 == true_idx)
            roi_top5_match = int(true_idx in roi_top5)

            roi_top1_correct += roi_top1_match
            roi_top5_correct += roi_top5_match
            roi_confidences.append(roi_conf)

        except:
            continue

        total += 1

        per_image_results.append(
            {
                "image": img_name,
                "true_species": ref["species"],
                "full_image_pred": species_map.get(int(full_top1), "unknown"),
                "full_image_conf": round(float(full_conf), 4),
                "full_image_correct": bool(full_top1_match),
                "roi_pred": species_map.get(int(roi_top1), "unknown"),
                "roi_conf": round(float(roi_conf), 4),
                "roi_correct": bool(roi_top1_match),
            }
        )

    if total == 0:
        print("No comparable images found!")
        return

    # Print comparison
    print(f"\n{'='*60}")
    print(f"FULL IMAGE vs CROPPED ROI COMPARISON")
    print(f"{'='*60}")
    print(f"  Images compared: {total}")
    print(f"")
    print(
        f"  {'Metric':<30} {'Full Image':>12} {'Cropped ROI':>12} {'Improvement':>12}"
    )
    print(f"  {'-'*66}")

    full_t1 = full_top1_correct / total
    roi_t1 = roi_top1_correct / total
    print(
        f"  {'Top-1 Accuracy':<30} {full_t1:>12.1%} {roi_t1:>12.1%} {roi_t1-full_t1:>+12.1%}"
    )

    full_t5 = full_top5_correct / total
    roi_t5 = roi_top5_correct / total
    print(
        f"  {'Top-5 Accuracy':<30} {full_t5:>12.1%} {roi_t5:>12.1%} {roi_t5-full_t5:>+12.1%}"
    )

    full_mc = np.mean(full_confidences)
    roi_mc = np.mean(roi_confidences)
    print(
        f"  {'Mean Confidence':<30} {full_mc:>12.3f} {roi_mc:>12.3f} {roi_mc-full_mc:>+12.3f}"
    )

    full_mdc = np.median(full_confidences)
    roi_mdc = np.median(roi_confidences)
    print(
        f"  {'Median Confidence':<30} {full_mdc:>12.3f} {roi_mdc:>12.3f} {roi_mdc-full_mdc:>+12.3f}"
    )

    # Save results
    comparison_table = {
        "Metric": [
            "Top-1 Accuracy",
            "Top-5 Accuracy",
            "Mean Confidence",
            "Median Confidence",
            "Images Compared",
        ],
        "Full_Image": [
            round(full_t1, 4),
            round(full_t5, 4),
            round(full_mc, 4),
            round(float(full_mdc), 4),
            total,
        ],
        "Cropped_ROI": [
            round(roi_t1, 4),
            round(roi_t5, 4),
            round(roi_mc, 4),
            round(float(roi_mdc), 4),
            total,
        ],
        "Improvement": [
            round(roi_t1 - full_t1, 4),
            round(roi_t5 - full_t5, 4),
            round(roi_mc - full_mc, 4),
            round(float(roi_mdc - full_mdc), 4),
            0,
        ],
    }

    df_comparison = pd.DataFrame(comparison_table)
    df_comparison.to_csv(output_dir / "12_full_vs_roi_comparison.csv", index=False)

    df_per_image = pd.DataFrame(per_image_results)
    df_per_image.to_csv(output_dir / "13_per_image_comparison.csv", index=False)

    print(f"\n  Saved to: {output_dir}/12_full_vs_roi_comparison.csv")
    print(f"  Saved to: {output_dir}/13_per_image_comparison.csv")

    return df_comparison


if __name__ == "__main__":
    from src.config import CONFIG, DEVICE

    csvs = CONFIG["paths"]["csvs"]
    compare_full_vs_roi(
        f"{csvs}/03_flower_rois.csv",
        f"{csvs}/08_final_labeled_dataset.csv",
        CONFIG["paths"]["raw_images"],
        CONFIG["paths"]["model_weights"],
        CONFIG["paths"]["class_idx_to_species"],
        CONFIG["paths"]["species_id_to_name"],
        csvs,
        device=str(DEVICE),
    )
