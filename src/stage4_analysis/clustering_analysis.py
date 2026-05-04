"""
Step 4.1 & 4.2: UMAP/t-SNE embeddings and clustering vs prediction agreement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import (
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)
from pathlib import Path

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from src.utils.plotting_utils import set_paper_style, save_figure


def run_clustering_analysis(
    clean_features_npz, clean_roi_ids_path, final_labels_csv, output_dir, config=None
):
    """
    Full clustering analysis pipeline:
    1. Dimensionality reduction (UMAP + t-SNE)
    2. Unsupervised clustering (DBSCAN)
    3. Compare clusters vs model predictions
    """
    set_paper_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load(clean_features_npz)
    with open(clean_roi_ids_path) as f:
        clean_roi_ids = [line.strip() for line in f if line.strip()]

    feature_matrix = np.array([data[k] for k in clean_roi_ids])
    feature_matrix = normalize(feature_matrix, norm="l2")

    df_labels = pd.read_csv(final_labels_csv)

    # Map ROI IDs to genus labels
    genus_labels = []
    for roi_id in clean_roi_ids:
        match = df_labels[df_labels["roi_id"] == roi_id]
        if len(match) > 0:
            genus_labels.append(
                match.iloc[0]["validated_genus"]
                if "validated_genus" in match.columns
                else match.iloc[0]["final_genus"]
            )
        else:
            genus_labels.append("Unknown")

    le = LabelEncoder()
    genus_encoded = le.fit_transform(genus_labels)

    # ── t-SNE ──
    perplexity = min(30, len(feature_matrix) - 1)
    if config:
        perplexity = min(
            config["visualization"]["tsne_perplexity"], len(feature_matrix) - 1
        )

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(feature_matrix)

    # ── UMAP ──
    X_umap = None
    if HAS_UMAP:
        print("Running UMAP...")
        n_neighbors = 15
        if config:
            n_neighbors = min(
                config["visualization"]["umap_n_neighbors"], len(feature_matrix) - 1
            )
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42
        )
        X_umap = reducer.fit_transform(feature_matrix)

    # ── DBSCAN Clustering ──
    embedding_for_cluster = X_umap if X_umap is not None else X_tsne
    eps = config["visualization"]["dbscan_eps"] if config else 0.5
    min_samples = config["visualization"]["dbscan_min_samples"] if config else 3

    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = db.fit_predict(embedding_for_cluster)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

    # ── Cluster vs Prediction Agreement ──
    mask = cluster_labels != -1
    agreement_stats = {}
    if mask.sum() > 0:
        nmi = normalized_mutual_info_score(genus_encoded[mask], cluster_labels[mask])
        h, c, v = homogeneity_completeness_v_measure(
            genus_encoded[mask], cluster_labels[mask]
        )
        agreement_stats = {
            "NMI": round(nmi, 4),
            "Homogeneity": round(h, 4),
            "Completeness": round(c, 4),
            "V_measure": round(v, 4),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }
        print(f"NMI: {nmi:.3f}, V-measure: {v:.3f}")

    # ── FIGURE 1: Embeddings colored by genus ──
    n_plots = 2 if X_umap is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    scatter = axes[0].scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=genus_encoded, cmap="tab20", s=30, alpha=0.7
    )
    axes[0].set_title("t-SNE of PlantNet Features\n(colored by predicted genus)")
    axes[0].set_xlabel("t-SNE-1")
    axes[0].set_ylabel("t-SNE-2")

    if X_umap is not None:
        axes[1].scatter(
            X_umap[:, 0], X_umap[:, 1], c=genus_encoded, cmap="tab20", s=30, alpha=0.7
        )
        axes[1].set_title("UMAP of PlantNet Features\n(colored by predicted genus)")
        axes[1].set_xlabel("UMAP-1")
        axes[1].set_ylabel("UMAP-2")

    unique_genera = list(set(genus_labels))
    if len(unique_genera) <= 20:
        cmap = plt.cm.tab20
        handles = [
            plt.scatter(
                [],
                [],
                c=[cmap(le.transform([g])[0] / max(len(unique_genera), 1))],
                label=g,
                s=50,
            )
            for g in sorted(unique_genera)[:15]
        ]
        fig.legend(
            handles=handles,
            loc="center right",
            fontsize=8,
            title="Genus",
            bbox_to_anchor=(1.15, 0.5),
        )

    plt.tight_layout()
    save_figure(fig, output_dir / "feature_space_embeddings.png")
    plt.close()

    # ── FIGURE 2: Clusters vs Predictions side by side ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    plot_embedding = X_umap if X_umap is not None else X_tsne
    embed_name = "UMAP" if X_umap is not None else "t-SNE"

    axes[0].scatter(
        plot_embedding[:, 0],
        plot_embedding[:, 1],
        c=genus_encoded,
        cmap="tab20",
        s=30,
        alpha=0.7,
    )
    axes[0].set_title(f"Colored by PlantNet Prediction (Genus)")

    axes[1].scatter(
        plot_embedding[:, 0],
        plot_embedding[:, 1],
        c=cluster_labels,
        cmap="tab20",
        s=30,
        alpha=0.7,
    )
    axes[1].set_title(f"Colored by DBSCAN Clusters")

    plt.suptitle(
        f"Do unsupervised feature clusters align with model predictions?\n"
        f'NMI={agreement_stats.get("NMI", "N/A")}, '
        f'V-measure={agreement_stats.get("V_measure", "N/A")}',
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, output_dir / "clusters_vs_predictions.png")
    plt.close()

    # Save stats
    pd.DataFrame([agreement_stats]).to_csv(
        output_dir.parent / "tables" / "clustering_agreement.csv", index=False
    )

    # Save embeddings for reuse
    np.savez(
        output_dir.parent.parent / "data" / "features" / "embeddings.npz",
        tsne=X_tsne,
        umap=X_umap if X_umap is not None else np.array([]),
        cluster_labels=cluster_labels,
        genus_encoded=genus_encoded,
    )

    return X_tsne, X_umap, cluster_labels, agreement_stats


if __name__ == "__main__":
    from src.config import CONFIG

    run_clustering_analysis(
        f"{CONFIG['paths']['features']}/clean_features.npz",
        f"{CONFIG['paths']['features']}/clean_roi_ids.txt",
        f"{CONFIG['paths']['csvs']}/08_final_labeled_dataset.csv",
        CONFIG["paths"]["figures"],
        CONFIG,
    )
