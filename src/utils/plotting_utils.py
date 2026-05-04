"""
Common plotting utilities and style settings.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np


def set_paper_style():
    """Set matplotlib style suitable for paper figures"""
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, filepath, dpi=300):
    """Save figure with standard settings"""
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {filepath}")


def create_image_grid(images, titles=None, ncols=4, figsize_per_image=3):
    """Create a grid of images"""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per_image * ncols, figsize_per_image * nrows)
    )
    axes = axes.flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig
