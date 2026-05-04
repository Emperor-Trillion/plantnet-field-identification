"""
Step 2.4: Visualize deduplication results for paper figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from src.utils.plotting_utils import set_paper_style, save_figure


def visualize_deduplication_results(
    sim_matrix, roi_ids, pairs_csv, threshold_csv, roi_dir, output_dir
):
    """Generate all deduplication-related figures"""
    set_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_pairs = pd.read_csv(pairs_csv)
    df_thresh = pd.read_csv(threshold_csv)

    # Figure 1: Similarity distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    ax.hist(upper_tri, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(
        x=0.95, color="red", linestyle="--", linewidth=2, label="Dedup threshold (0.95)"
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Number of ROI Pairs")
    ax.set_title("Distribution of Pairwise Feature Similarities")
    ax.legend()
    save_figure(fig, output_dir / "similarity_distribution.png")
    plt.close()

    # Figure 2: Threshold sensitivity curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df_thresh["threshold"],
        df_thresh["unique_rois_remaining"],
        "bo-",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Unique ROIs Remaining")
    ax.set_title("Feature Deduplication: Threshold Sensitivity")
    ax.grid(True, alpha=0.3)
    for _, row in df_thresh.iterrows():
        ax.annotate(
            f"{int(row['unique_rois_remaining'])}",
            (row["threshold"], row["unique_rois_remaining"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    save_figure(fig, output_dir / "threshold_sensitivity.png")
    plt.close()

    # Figure 3: Example duplicate pairs
    if len(df_pairs) > 0:
        num_examples = min(9, len(df_pairs))
        fig, axes = plt.subplots(num_examples, 2, figsize=(8, 3 * num_examples))
        fig.suptitle(
            "Feature-Level Duplicates Detected", fontsize=14, fontweight="bold"
        )

        if num_examples == 1:
            axes = [axes]

        for idx in range(num_examples):
            pair = df_pairs.iloc[idx]
            try:
                img_a = Image.open(f"{roi_dir}/{pair['roi_a']}.jpg")
                img_b = Image.open(f"{roi_dir}/{pair['roi_b']}.jpg")

                axes[idx][0].imshow(img_a)
                axes[idx][0].set_title(f"{pair['roi_a']}", fontsize=8)
                axes[idx][0].axis("off")

                axes[idx][1].imshow(img_b)
                axes[idx][1].set_title(
                    f"{pair['roi_b']} (sim={pair['similarity']})", fontsize=8
                )
                axes[idx][1].axis("off")
            except:
                pass

        plt.tight_layout()
        save_figure(fig, output_dir / "duplicate_examples.png")
        plt.close()

    print(f"Deduplication figures saved to: {output_dir}")


if __name__ == "__main__":
    from src.config import CONFIG
    from src.stage2_deduplication.similarity_matrix import compute_similarity_matrix

    roi_ids, _, sim_matrix = compute_similarity_matrix(
        f"{CONFIG['paths']['features']}/roi_features.npz"
    )

    visualize_deduplication_results(
        sim_matrix,
        roi_ids,
        f"{CONFIG['paths']['csvs']}/05_feature_duplicates.csv",
        f"{CONFIG['paths']['csvs']}/05b_threshold_analysis.csv",
        CONFIG["paths"]["cropped_rois"],
        CONFIG["paths"]["figures"],
    )
