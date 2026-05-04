"""
Step 4.4: Comprehensive statistics and paper-ready charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils.plotting_utils import set_paper_style, save_figure


def generate_all_statistics(
    master_csv,
    rois_csv,
    classification_csv,
    final_labels_csv,
    dedup_threshold_csv,
    output_fig_dir,
    output_table_dir,
):
    """Generate the comprehensive 9-panel statistics figure and summary tables"""
    set_paper_style()
    output_fig_dir = Path(output_fig_dir)
    output_table_dir = Path(output_table_dir)
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    output_table_dir.mkdir(parents=True, exist_ok=True)

    df_master = pd.read_csv(master_csv)
    # Filter to images that actually exist on disk. The master table preserves
    # pre-curation entries (303 raw images) as a record of the curation step;
    # downstream statistics should reflect only the 202 images retained for
    # the pipeline. See paper §4.2 for curation criteria.
    df_master = df_master[
        df_master["file_path"].apply(lambda p: Path(p).exists())
    ].reset_index(drop=True)
    df_rois = pd.read_csv(rois_csv)
    df_class = pd.read_csv(classification_csv)
    df_final = pd.read_csv(final_labels_csv)
    df_thresh = pd.read_csv(dedup_threshold_csv)

    # Determine genus column name
    genus_col = (
        "validated_genus" if "validated_genus" in df_final.columns else "final_genus"
    )

    # ── 9-PANEL COMPREHENSIVE FIGURE ──
    fig = plt.figure(figsize=(24, 20))

    # Panel 1: Top genera distribution
    ax1 = fig.add_subplot(3, 3, 1)
    top_genera = df_final[genus_col].value_counts().head(15)
    top_genera.plot(kind="barh", ax=ax1, color="forestgreen", edgecolor="black")
    ax1.set_title("Top 15 Genera Identified")
    ax1.set_xlabel("Count")

    # Panel 2: Confidence tier distribution
    ax2 = fig.add_subplot(3, 3, 2)
    if "confidence_tier" in df_class.columns:
        tier_counts = df_class["confidence_tier"].value_counts()
        colors = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
        tier_counts.plot(
            kind="bar", ax=ax2, color=colors[: len(tier_counts)], edgecolor="black"
        )
        ax2.set_title("Prediction Confidence Tiers")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=45)

    # Panel 3: Confidence score distribution
    ax3 = fig.add_subplot(3, 3, 3)
    if "top1_confidence" in df_class.columns:
        ax3.hist(
            df_class["top1_confidence"],
            bins=30,
            edgecolor="black",
            alpha=0.7,
            color="steelblue",
        )
        median_conf = df_class["top1_confidence"].median()
        ax3.axvline(
            x=median_conf,
            color="red",
            linestyle="--",
            label=f"Median: {median_conf:.2f}",
        )
        ax3.set_title("Top-1 Confidence Distribution")
        ax3.set_xlabel("Confidence")
        ax3.set_ylabel("Count")
        ax3.legend()

    # Panel 4: Scene category pie chart
    ax4 = fig.add_subplot(3, 3, 4)
    if "scene_category" in df_master.columns:
        scene_counts = df_master["scene_category"].value_counts()
        scene_counts.plot(kind="pie", ax=ax4, autopct="%1.1f%%", fontsize=8)
        ax4.set_title("Image Scene Categories")
        ax4.set_ylabel("")

    # Panel 5: ROIs per image histogram
    ax5 = fig.add_subplot(3, 3, 5)
    rois_per_img = df_rois.groupby("source_image").size()
    ax5.hist(
        rois_per_img,
        bins=range(0, int(rois_per_img.max()) + 2),
        edgecolor="black",
        alpha=0.7,
        color="mediumpurple",
    )
    ax5.set_title("Flower ROIs per Image")
    ax5.set_xlabel("Number of ROIs")
    ax5.set_ylabel("Number of Images")

    # Panel 6: Deduplication threshold sensitivity
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(
        df_thresh["threshold"],
        df_thresh["unique_rois_remaining"],
        "bo-",
        linewidth=2,
        markersize=8,
    )
    ax6.set_title("Deduplication Threshold Sensitivity")
    ax6.set_xlabel("Similarity Threshold")
    ax6.set_ylabel("Unique ROIs Remaining")
    ax6.grid(True, alpha=0.3)
    for _, row in df_thresh.iterrows():
        ax6.annotate(
            f"{int(row['unique_rois_remaining'])}",
            (row["threshold"], row["unique_rois_remaining"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    # Panel 7: Label source breakdown
    ax7 = fig.add_subplot(3, 3, 7)
    source_col = (
        "validation_method"
        if "validation_method" in df_final.columns
        else "label_source"
    )
    if source_col in df_final.columns:
        df_final[source_col].value_counts().plot(
            kind="bar", ax=ax7, color="coral", edgecolor="black"
        )
        ax7.set_title("How Labels Were Assigned")
        ax7.tick_params(axis="x", rotation=45)

    # Panel 8: Genus agreement
    ax8 = fig.add_subplot(3, 3, 8)
    if "genus_agreement" in df_class.columns:
        agree_counts = df_class["genus_agreement"].value_counts()
        agree_counts.index = [
            "Different Genus" if not x else "Same Genus" for x in agree_counts.index
        ]
        agree_counts.plot(
            kind="bar", ax=ax8, color=["#e74c3c", "#2ecc71"], edgecolor="black"
        )
        ax8.set_title("Top-1 & Top-2 Same Genus?")
        ax8.tick_params(axis="x", rotation=0)

    # Panel 9: Confidence gap distribution
    ax9 = fig.add_subplot(3, 3, 9)
    if "top1_top2_gap" in df_class.columns:
        ax9.hist(
            df_class["top1_top2_gap"],
            bins=30,
            edgecolor="black",
            alpha=0.7,
            color="mediumpurple",
        )
        ax9.set_title("Confidence Gap (Top-1 minus Top-2)")
        ax9.set_xlabel("Gap")
        ax9.set_ylabel("Count")

    plt.tight_layout()
    save_figure(fig, output_fig_dir / "comprehensive_statistics.png")
    plt.close()

    # ── SUMMARY TABLES ──

    # Table: Dataset overview
    dataset_summary = {
        "Metric": [
            "Curated Raw Images (post-quality-filter)",
            "Total ROIs Extracted",
            "ROIs After Deduplication",
            "Unique Species Predicted",
            "Unique Genera Predicted",
            "Images with Humans",
            "Images with Multiple Flower Types",
            "Mean Confidence Score",
            "Median Confidence Score",
        ],
        "Value": [
            len(df_master),
            len(df_rois),
            len(df_class),
            (
                df_class["pred1_species"].nunique()
                if "pred1_species" in df_class.columns
                else "N/A"
            ),
            df_final[genus_col].nunique(),
            (
                df_master["yolo_persons"].gt(0).sum()
                if "yolo_persons" in df_master.columns
                else "N/A"
            ),
            (
                df_master["num_flower_rois"].gt(1).sum()
                if "num_flower_rois" in df_master.columns
                else "N/A"
            ),
            (
                round(df_class["top1_confidence"].mean(), 3)
                if "top1_confidence" in df_class.columns
                else "N/A"
            ),
            (
                round(df_class["top1_confidence"].median(), 3)
                if "top1_confidence" in df_class.columns
                else "N/A"
            ),
        ],
    }
    df_summary = pd.DataFrame(dataset_summary)
    df_summary.to_csv(output_table_dir / "dataset_summary.csv", index=False)

    print(f"\nStatistics figures saved to: {output_fig_dir}")
    print(f"Summary tables saved to: {output_table_dir}")
    print(f"\nDataset Summary:")
    print(df_summary.to_string(index=False))

    return df_summary


if __name__ == "__main__":
    from src.config import CONFIG

    csvs = CONFIG["paths"]["csvs"]
    generate_all_statistics(
        f"{csvs}/04_master_image_table.csv",
        f"{csvs}/03_flower_rois.csv",
        f"{csvs}/06_classification_results.csv",
        f"{csvs}/08_final_labeled_dataset.csv",
        f"{csvs}/05b_threshold_analysis.csv",
        CONFIG["paths"]["figures"],
        f"{CONFIG['paths']['tables'] if 'tables' in CONFIG['paths'] else 'outputs/tables'}",
    )
