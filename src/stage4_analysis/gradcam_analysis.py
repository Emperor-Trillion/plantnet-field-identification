"""
Step 4.3: Grad-CAM attention visualizations.
Shows what the PlantNet model focuses on when making predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("WARNING: pytorch-grad-cam not installed. Grad-CAM figures will be skipped.")

from src.utils.model_utils import load_plantnet_model, load_species_mapping
from src.utils.image_utils import load_image_pil, get_preprocess_transform
from src.utils.plotting_utils import set_paper_style, save_figure


def generate_single_gradcam(
    image_path, model, cam, preprocess, species_map, device="cpu"
):
    """Generate Grad-CAM for one image"""
    img = load_image_pil(image_path)
    img_resized = img.resize((224, 224))
    rgb_img = np.array(img_resized) / 255.0

    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_conf = F.softmax(output, dim=1)[0][pred_class].item()

    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

    pred_name = species_map.get(pred_class, f"class_{pred_class}")
    return visualization, rgb_img, pred_name, pred_conf


def run_gradcam_analysis(
    classification_csv,
    roi_dir,
    weights_path,
    class_idx_path,
    species_id_to_name_path,
    output_dir,
    device="cpu",
):
    """Generate Grad-CAM figures organized by confidence tier"""
    if not HAS_GRADCAM:
        print("Skipping Grad-CAM analysis (pytorch-grad-cam not installed)")
        return

    set_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_plantnet_model(weights_path, class_idx_path, device=device)
    species_map = load_species_mapping(class_idx_path, species_id_to_name_path)
    preprocess = get_preprocess_transform()

    model_name = type(model).__name__
    if "ResNet" in model_name:
        target_layers = [model.layer4[-1]]
    elif "EfficientNet" in model_name:
        # timm EfficientNet
        target_layers = [model.conv_head]
    elif "DenseNet" in model_name:
        target_layers = [model.features.denseblock4]
    else:
        target_layers = [list(model.children())[-2]]
    cam = GradCAM(model=model, target_layers=target_layers)

    df_class = pd.read_csv(classification_csv)

    # ── FIGURE 1: Grad-CAM by confidence tier ──
    tiers = ["TIER_1_HIGH", "TIER_2_GOOD", "TIER_3_MODERATE", "TIER_4_LOW"]
    samples_per_tier = 4

    fig, axes = plt.subplots(
        len(tiers), samples_per_tier * 2, figsize=(samples_per_tier * 5, len(tiers) * 3)
    )
    fig.suptitle(
        "Grad-CAM: What Does PlantNet Focus On?\n"
        "(Left: Original | Right: Attention Map)",
        fontsize=14,
        fontweight="bold",
    )

    for tier_idx, tier in enumerate(tiers):
        tier_rois = df_class[df_class["confidence_tier"] == tier].head(samples_per_tier)

        for col_idx, (_, row) in enumerate(tier_rois.iterrows()):
            if col_idx >= samples_per_tier:
                break

            roi_path = f"{roi_dir}/{row['roi_id']}.jpg"
            orig_col = col_idx * 2
            cam_col = col_idx * 2 + 1

            try:
                vis, rgb_img, pred_name, pred_conf = generate_single_gradcam(
                    roi_path, model, cam, preprocess, species_map, device
                )

                axes[tier_idx][orig_col].imshow(rgb_img)
                axes[tier_idx][orig_col].set_title(f'{row["roi_id"]}', fontsize=7)
                axes[tier_idx][orig_col].axis("off")

                axes[tier_idx][cam_col].imshow(vis)
                axes[tier_idx][cam_col].set_title(
                    f"{pred_name[:20]}\nConf: {pred_conf:.2f}", fontsize=7
                )
                axes[tier_idx][cam_col].axis("off")
            except Exception as e:
                axes[tier_idx][orig_col].text(0.5, 0.5, "Error", ha="center")
                axes[tier_idx][orig_col].axis("off")
                axes[tier_idx][cam_col].axis("off")

        axes[tier_idx][0].set_ylabel(
            tier.replace("_", "\n"),
            fontsize=9,
            fontweight="bold",
            rotation=0,
            labelpad=60,
        )

    plt.tight_layout()
    save_figure(fig, output_dir / "gradcam_by_tier.png")
    plt.close()

    # ── FIGURE 2: Success vs Failure cases ──
    high_conf = df_class[df_class["top1_confidence"] > 0.8].head(4)
    low_conf = df_class[df_class["top1_confidence"] < 0.3].head(4)

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    fig.suptitle(
        "Grad-CAM: High Confidence (top) vs Low Confidence (bottom)",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, subset in enumerate([high_conf, low_conf]):
        for col_idx, (_, row) in enumerate(subset.iterrows()):
            if col_idx >= 4:
                break
            roi_path = f"{roi_dir}/{row['roi_id']}.jpg"
            try:
                vis, rgb_img, pred_name, pred_conf = generate_single_gradcam(
                    roi_path, model, cam, preprocess, species_map, device
                )
                axes[row_idx][col_idx * 2].imshow(rgb_img)
                axes[row_idx][col_idx * 2].axis("off")
                axes[row_idx][col_idx * 2 + 1].imshow(vis)
                axes[row_idx][col_idx * 2 + 1].set_title(
                    f"{pred_name[:18]}\n{pred_conf:.2f}", fontsize=7
                )
                axes[row_idx][col_idx * 2 + 1].axis("off")
            except:
                axes[row_idx][col_idx * 2].axis("off")
                axes[row_idx][col_idx * 2 + 1].axis("off")

    plt.tight_layout()
    save_figure(fig, output_dir / "gradcam_success_vs_failure.png")
    plt.close()

    print(f"Grad-CAM figures saved to {output_dir}")


if __name__ == "__main__":
    from src.config import CONFIG, DEVICE

    run_gradcam_analysis(
        f"{CONFIG['paths']['csvs']}/06_classification_results.csv",
        CONFIG["paths"]["cropped_rois"],
        CONFIG["paths"]["model_weights"],
        CONFIG["paths"]["class_idx_to_species"],
        CONFIG["paths"]["species_id_to_name"],
        CONFIG["paths"]["figures"],
        device=str(DEVICE),
    )
