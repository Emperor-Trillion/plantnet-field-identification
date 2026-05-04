"""
Step 4.5: Analysis of images containing multiple flower species.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image
from pathlib import Path
from src.utils.plotting_utils import set_paper_style, save_figure


def run_multi_species_analysis(
    master_csv,
    rois_csv,
    final_labels_csv,
    raw_image_dir,
    output_fig_dir,
    output_table_dir,
):
    """Analyze and visualize images with multiple flower species"""
    set_paper_style()
    output_fig_dir = Path(output_fig_dir)
    output_table_dir = Path(output_table_dir)
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    output_table_dir.mkdir(parents=True, exist_ok=True)

    df_master = pd.read_csv(master_csv)
    df_rois = pd.read_csv(rois_csv)
    df_final = pd.read_csv(final_labels_csv)

    genus_col = (
        "validated_genus" if "validated_genus" in df_final.columns else "final_genus"
    )
    species_col = (
        "validated_species"
        if "validated_species" in df_final.columns
        else "final_species"
    )

    # ── Find multi-species images ──
    multi_species = []

    for img_id in df_rois["source_image"].unique():
        img_rois = df_rois[df_rois["source_image"] == img_id]
        img_labels = df_final[df_final["source_image"] == img_id]

        if len(img_rois) < 2:
            continue

        genera = img_labels[genus_col].unique() if len(img_labels) > 0 else []
        species = img_labels[species_col].unique() if len(img_labels) > 0 else []

        multi_species.append(
            {
                "image_id": img_id,
                "total_rois": len(img_rois),
                "unique_genera": len(genera),
                "unique_species": len(species),
                "genera_list": ", ".join(genera),
                "species_list": ", ".join(species),
                "is_multi_genus": len(genera) > 1,
                "is_multi_species": len(species) > 1,
            }
        )

    df_multi = pd.DataFrame(multi_species)
    df_multi.to_csv(output_table_dir / "multi_species_images.csv", index=False)

    print(f"\nMulti-Species Analysis:")
    print(f"  Images with 2+ ROIs: {len(df_multi)}")
    print(f"  Images with 2+ genera: {df_multi['is_multi_genus'].sum()}")
    print(f"  Images with 2+ species: {df_multi['is_multi_species'].sum()}")

    # ── FIGURE 1: Multi-species examples with bounding boxes ──
    multi_genus_images = df_multi[df_multi["is_multi_genus"] == True]
    examples = (
        multi_genus_images.head(4) if len(multi_genus_images) > 0 else df_multi.head(4)
    )

    if len(examples) > 0:
        fig, axes_grid = plt.subplots(len(examples), 1, figsize=(14, 5 * len(examples)))
        if len(examples) == 1:
            axes_grid = [axes_grid]

        fig.suptitle(
            "Multi-Species Image Detection Results", fontsize=14, fontweight="bold"
        )

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, (_, img_row) in enumerate(examples.iterrows()):
            img_id = img_row["image_id"]
            img_path = Path(raw_image_dir) / img_id

            if not img_path.exists():
                axes_grid[idx].text(0.5, 0.5, f"Image not found: {img_id}", ha="center")
                continue

            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes_grid[idx].imshow(img_rgb)

            img_rois = df_rois[df_rois["source_image"] == img_id]
            img_labels = df_final[df_final["source_image"] == img_id]

            genus_to_color = {}
            color_idx = 0

            for _, roi_row in img_rois.iterrows():
                roi_id = roi_row["roi_id"]
                label_match = img_labels[img_labels["roi_id"] == roi_id]

                genus = "Unknown"
                species = "Unknown"
                if len(label_match) > 0:
                    genus = label_match.iloc[0][genus_col]
                    species = label_match.iloc[0][species_col]

                if genus not in genus_to_color:
                    genus_to_color[genus] = colors[color_idx % len(colors)]
                    color_idx += 1

                color = genus_to_color[genus]
                x1 = roi_row["bbox_x1"]
                y1 = roi_row["bbox_y1"]
                w = roi_row["bbox_x2"] - roi_row["bbox_x1"]
                h = roi_row["bbox_y2"] - roi_row["bbox_y1"]

                rect = mpatches.FancyBboxPatch(
                    (x1, y1),
                    w,
                    h,
                    boxstyle="round,pad=3",
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                )
                axes_grid[idx].add_patch(rect)
                axes_grid[idx].text(
                    x1,
                    y1 - 5,
                    f"{genus}",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

            axes_grid[idx].set_title(f'{img_id}: {img_row["genera_list"]}', fontsize=10)
            axes_grid[idx].axis("off")

        plt.tight_layout()
        save_figure(fig, output_fig_dir / "multi_species_examples.png")
        plt.close()

    # ── FIGURE 2: Distribution of species count per image ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(
        df_multi["unique_genera"],
        bins=range(1, df_multi["unique_genera"].max() + 2),
        edgecolor="black",
        alpha=0.7,
        color="forestgreen",
    )
    axes[0].set_title("Unique Genera per Multi-ROI Image")
    axes[0].set_xlabel("Number of Genera")
    axes[0].set_ylabel("Number of Images")

    axes[1].hist(
        df_multi["total_rois"],
        bins=range(1, df_multi["total_rois"].max() + 2),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    axes[1].set_title("Total ROIs per Multi-ROI Image")
    axes[1].set_xlabel("Number of ROIs")
    axes[1].set_ylabel("Number of Images")

    plt.tight_layout()
    save_figure(fig, output_fig_dir / "multi_species_distribution.png")
    plt.close()

    print(f"Multi-species figures saved to {output_fig_dir}")
    return df_multi


if __name__ == "__main__":
    from src.config import CONFIG

    csvs = CONFIG["paths"]["csvs"]
    run_multi_species_analysis(
        f"{csvs}/04_master_image_table.csv",
        f"{csvs}/03_flower_rois.csv",
        f"{csvs}/08_final_labeled_dataset.csv",
        CONFIG["paths"]["raw_images"],
        CONFIG["paths"]["figures"],
        "outputs/tables",
    )
