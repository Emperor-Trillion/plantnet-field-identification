"""
Step 2.2: Compute pairwise similarity matrix from features.
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


def compute_similarity_matrix(features_npz_path, output_dir=None):
    """
    Load features and compute full pairwise cosine similarity matrix.

    Returns:
        roi_ids: list of ROI identifiers
        feature_matrix: normalized feature matrix (N, 2048)
        sim_matrix: pairwise cosine similarity (N, N)
    """
    data = np.load(features_npz_path)
    roi_ids = sorted(list(data.keys()))
    feature_matrix = np.array([data[k] for k in roi_ids])

    # L2 normalize
    feature_matrix_norm = normalize(feature_matrix, norm="l2")

    # Compute similarity
    sim_matrix = cosine_similarity(feature_matrix_norm)

    print(f"Feature matrix shape: {feature_matrix_norm.shape}")
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Similarity range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")

    # Get upper triangle stats (excluding diagonal)
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    print(f"Mean pairwise similarity: {upper_tri.mean():.4f}")
    print(f"Median pairwise similarity: {np.median(upper_tri):.4f}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "similarity_matrix.npy", sim_matrix)
        np.save(output_dir / "feature_matrix_normalized.npy", feature_matrix_norm)
        with open(output_dir / "roi_ids_order.txt", "w") as f:
            for rid in roi_ids:
                f.write(f"{rid}\n")

    return roi_ids, feature_matrix_norm, sim_matrix


if __name__ == "__main__":
    from src.config import CONFIG

    compute_similarity_matrix(
        f"{CONFIG['paths']['features']}/roi_features.npz", CONFIG["paths"]["features"]
    )
