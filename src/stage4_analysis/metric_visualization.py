"""
src/stage4_analysis/metric_visualization.py

Generate paper-ready figures for metric results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plotting_utils import set_paper_style, save_figure


def plot_all_metric_figures(
    metrics_csv, genus_csv, per_class_detail, y_true, y_prob, output_dir
):
    """Generate all metric-related figures for the paper"""
    set_paper_style()

    df_metrics = pd.read_csv(metrics_csv)
    df_genus = pd.read_csv(genus_csv, index_col=0)

    # ──────────────────────────────────────────
    # FIGURE 1: Top-k Accuracy Curve
    # "How accuracy improves as we allow more guesses"
    # ──────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 5))

    k_values = [1, 2, 3, 4, 5]
    micro_accs = [
        df_metrics[df_metrics["Metric"] == f"Top-{k} Accuracy"]["Value"].values[0]
        for k in k_values
    ]

    # If you have macro values for all k
    macro_accs = []
    for k in k_values:
        row = df_metrics[df_metrics["Metric"] == f"Macro Top-{k} Accuracy"]
        if len(row) > 0:
            macro_accs.append(row["Value"].values[0])

    ax.plot(
        k_values,
        micro_accs,
        "bo-",
        linewidth=2,
        markersize=8,
        label="Top-k Accuracy (micro)",
    )

    if len(macro_accs) == len(k_values):
        ax.plot(
            k_values,
            macro_accs,
            "rs--",
            linewidth=2,
            markersize=8,
            label="Macro Top-k Accuracy",
        )

    # Annotate each point
    for i, k in enumerate(k_values):
        ax.annotate(
            f"{micro_accs[i]:.1%}",
            (k, micro_accs[i]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
            color="blue",
        )

    ax.set_xlabel("k (number of allowed predictions)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-k Accuracy: How Many Guesses Does the Model Need?")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, f"{output_dir}/metric_topk_curve.png")
    plt.close()

    # ──────────────────────────────────────────
    # FIGURE 2: Micro vs Macro Gap
    # "Does the model favor common species over rare ones?"
    # ──────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(k_values))
    width = 0.35

    ax.bar(
        x - width / 2,
        micro_accs,
        width,
        label="Micro (sample-weighted)",
        color="steelblue",
        edgecolor="black",
    )

    if len(macro_accs) == len(k_values):
        ax.bar(
            x + width / 2,
            macro_accs,
            width,
            label="Macro (class-weighted)",
            color="coral",
            edgecolor="black",
        )

    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Micro vs Macro Top-k Accuracy\n"
        "(Large gap = model struggles with rare species)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    save_figure(fig, f"{output_dir}/metric_micro_vs_macro.png")
    plt.close()

    # ──────────────────────────────────────────
    # FIGURE 3: Per-Genus Performance
    # "Which flower genera are easy/hard to identify?"
    # ──────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Sort by top-1 accuracy
    df_genus_sorted = df_genus.sort_values("top_1_acc", ascending=True)

    # Left: Top-1 accuracy per genus
    colors = [
        "#e74c3c" if acc < 0.5 else "#f39c12" if acc < 0.8 else "#2ecc71"
        for acc in df_genus_sorted["top_1_acc"]
    ]

    axes[0].barh(
        range(len(df_genus_sorted)),
        df_genus_sorted["top_1_acc"],
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )
    axes[0].set_yticks(range(len(df_genus_sorted)))
    axes[0].set_yticklabels(df_genus_sorted.index, fontsize=8)
    axes[0].set_xlabel("Top-1 Accuracy")
    axes[0].set_title("Per-Genus Top-1 Accuracy")
    axes[0].axvline(x=0.5, color="red", linestyle=":", alpha=0.5)

    # Right: Sample count per genus
    df_genus_by_count = df_genus.sort_values("n_samples", ascending=True)
    axes[1].barh(
        range(len(df_genus_by_count)),
        df_genus_by_count["n_samples"],
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    axes[1].set_yticks(range(len(df_genus_by_count)))
    axes[1].set_yticklabels(df_genus_by_count.index, fontsize=8)
    axes[1].set_xlabel("Number of Samples")
    axes[1].set_title("Samples per Genus")

    plt.tight_layout()
    save_figure(fig, f"{output_dir}/metric_per_genus.png")
    plt.close()

    # ──────────────────────────────────────────
    # FIGURE 4: Confidence vs Correctness
    # "When the model is confident, is it actually right?"
    # ──────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 5))

    top1_preds = np.argmax(y_prob, axis=1)
    top1_confs = np.max(y_prob, axis=1)
    is_correct = top1_preds == y_true

    # Bin by confidence
    bins = np.arange(0, 1.05, 0.1)
    bin_accs = []
    bin_counts = []
    bin_centers = []

    for i in range(len(bins) - 1):
        mask = (top1_confs >= bins[i]) & (top1_confs < bins[i + 1])
        if mask.sum() > 0:
            bin_accs.append(is_correct[mask].mean())
            bin_counts.append(mask.sum())
            bin_centers.append((bins[i] + bins[i + 1]) / 2)

    # Reliability diagram
    ax.bar(
        bin_centers,
        bin_accs,
        width=0.08,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Actual accuracy",
    )
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")

    ax.set_xlabel("Model Confidence")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title("Reliability Diagram: Is the Model Well-Calibrated?")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # Add sample counts as text
    for center, count in zip(bin_centers, bin_counts):
        ax.text(center, -0.07, f"n={count}", ha="center", fontsize=7)

    save_figure(fig, f"{output_dir}/metric_reliability_diagram.png")
    plt.close()

    print(f"All metric figures saved to {output_dir}")
