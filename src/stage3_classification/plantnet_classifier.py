"""
Step 3.1: Zero-shot classification using PlantNet-300K.
Updated for .tar weights and two-file species mapping.
"""

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from src.utils.model_utils import load_plantnet_model, load_species_mapping
from src.utils.image_utils import load_image_pil, get_preprocess_transform


def classify_single_roi(
    image_path, model, preprocess, species_map, device="cpu", top_k=5
):
    """Classify one ROI image"""
    img = load_image_pil(image_path)
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(top_k)

    predictions = []
    for rank in range(top_k):
        class_idx = top_indices[0][rank].item()
        confidence = top_probs[0][rank].item()
        species_name = species_map.get(class_idx, f"unknown_{class_idx}")
        genus_name = species_name.split()[0] if " " in species_name else species_name

        predictions.append(
            {
                "rank": rank + 1,
                "class_index": class_idx,
                "species": species_name,
                "genus": genus_name,
                "confidence": round(confidence, 4),
            }
        )

    return predictions


def classify_all_rois(
    clean_roi_ids_path,
    roi_csv,
    weights_path,
    class_idx_path,
    species_id_to_name_path,
    output_csv,
    device="cpu",
    top_k=5,
):
    """Run PlantNet classification on all deduplicated ROIs"""

    with open(clean_roi_ids_path) as f:
        clean_roi_ids = [line.strip() for line in f if line.strip()]

    df_rois = pd.read_csv(roi_csv)
    model = load_plantnet_model(weights_path, class_idx_path, device)
    species_map = load_species_mapping(class_idx_path, species_id_to_name_path)
    preprocess = get_preprocess_transform()

    results = []

    for roi_id in tqdm(clean_roi_ids, desc="Classifying ROIs"):
        roi_row = df_rois[df_rois["roi_id"] == roi_id]
        if len(roi_row) == 0:
            continue

        roi_path = roi_row.iloc[0]["crop_path"]
        source_image = roi_row.iloc[0]["source_image"]

        try:
            preds = classify_single_roi(
                roi_path, model, preprocess, species_map, device, top_k
            )
        except Exception as e:
            print(f"Error classifying {roi_id}: {e}")
            continue

        result = {
            "roi_id": roi_id,
            "source_image": source_image,
        }

        for pred in preds:
            r = pred["rank"]
            result[f"pred{r}_species"] = pred["species"]
            result[f"pred{r}_genus"] = pred["genus"]
            result[f"pred{r}_conf"] = pred["confidence"]
            result[f"pred{r}_class_idx"] = pred["class_index"]

        result["top1_confidence"] = preds[0]["confidence"]
        result["top1_top2_gap"] = preds[0]["confidence"] - preds[1]["confidence"]
        result["top1_genus"] = preds[0]["genus"]
        result["top2_genus"] = preds[1]["genus"]
        result["genus_agreement"] = preds[0]["genus"] == preds[1]["genus"]

        results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)

    print(f"\nClassification Summary:")
    print(f"  ROIs classified: {len(df_results)}")
    print(f"  Unique species (top-1): {df_results['pred1_species'].nunique()}")
    print(f"  Unique genera (top-1): {df_results['pred1_genus'].nunique()}")
    print(f"  Mean confidence: {df_results['top1_confidence'].mean():.3f}")
    print(f"  Median confidence: {df_results['top1_confidence'].median():.3f}")

    return df_results


if __name__ == "__main__":
    from src.config import CONFIG, DEVICE

    classify_all_rois(
        f"{CONFIG['paths']['features']}/clean_roi_ids.txt",
        f"{CONFIG['paths']['csvs']}/03_flower_rois.csv",
        CONFIG["paths"]["model_weights"],
        CONFIG["paths"]["class_idx_to_species"],
        CONFIG["paths"]["species_id_to_name"],
        f"{CONFIG['paths']['csvs']}/06_classification_results.csv",
        device=str(DEVICE),
    )
