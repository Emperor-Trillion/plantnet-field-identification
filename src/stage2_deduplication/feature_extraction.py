"""
Step 2.1: Extract PlantNet deep features for every ROI.
Updated for .tar weights and two-file species mapping.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils.model_utils import get_feature_extractor
from src.utils.image_utils import load_image_pil, get_preprocess_transform


def extract_all_features(
    roi_csv,
    weights_path,
    class_idx_path,
    output_npz,
    also_extract_full_images=False,
    full_image_dir=None,
    full_output_npz=None,
    device="cpu",
):
    """Extract 2048-dim PlantNet features for all ROIs"""

    model, extractor = get_feature_extractor(weights_path, class_idx_path, device)
    preprocess = get_preprocess_transform()

    df_rois = pd.read_csv(roi_csv)

    roi_features = {}
    failed = []

    for _, row in tqdm(
        df_rois.iterrows(), total=len(df_rois), desc="Extracting ROI features"
    ):
        roi_id = row["roi_id"]
        roi_path = row["crop_path"]

        try:
            img = load_image_pil(roi_path)
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = extractor(input_tensor)

            roi_features[roi_id] = feat.squeeze().cpu().numpy()
        except Exception as e:
            failed.append({"roi_id": roi_id, "error": str(e)})

    Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **roi_features)

    print(f"\nFeature Extraction Summary:")
    print(f"  Successful: {len(roi_features)}")
    print(f"  Failed: {len(failed)}")
    sample_key = list(roi_features.keys())[0]
    actual_dim = roi_features[sample_key].shape[0]
    print(f"  Feature dim: {actual_dim}")
    print(f"  Saved to: {output_npz}")

    if also_extract_full_images and full_image_dir and full_output_npz:
        from src.utils.image_utils import get_image_files

        full_features = {}
        image_files = get_image_files(full_image_dir)

        for img_path in tqdm(image_files, desc="Extracting full image features"):
            try:
                img = load_image_pil(img_path)
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = extractor(input_tensor)
                full_features[img_path.stem] = feat.squeeze().cpu().numpy()
            except:
                pass

        np.savez(full_output_npz, **full_features)
        print(f"  Full image features saved to: {full_output_npz}")

    return roi_features, failed


if __name__ == "__main__":
    from src.config import CONFIG, DEVICE

    extract_all_features(
        f"{CONFIG['paths']['csvs']}/03_flower_rois.csv",
        CONFIG["paths"]["model_weights"],
        CONFIG["paths"]["class_idx_to_species"],
        f"{CONFIG['paths']['features']}/roi_features.npz",
        also_extract_full_images=True,
        full_image_dir=CONFIG["paths"]["raw_images"],
        full_output_npz=f"{CONFIG['paths']['features']}/full_image_features.npz",
        device=str(DEVICE),
    )
