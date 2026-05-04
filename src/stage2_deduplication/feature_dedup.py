"""
Step 2.3: Feature-level duplicate detection and removal.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def find_feature_duplicates(sim_matrix, roi_ids, threshold=0.95):
    """
    Identify duplicate/near-duplicate ROIs based on feature similarity.
    """
    n = len(roi_ids)
    duplicate_pairs = []
    duplicate_groups = {}
    is_duplicate = set()

    for i in range(n):
        if roi_ids[i] in is_duplicate:
            continue
        for j in range(i + 1, n):
            if roi_ids[j] in is_duplicate:
                continue
            sim = sim_matrix[i][j]
            if sim >= threshold:
                duplicate_pairs.append(
                    {
                        "roi_a": roi_ids[i],
                        "roi_b": roi_ids[j],
                        "similarity": round(float(sim), 4),
                        "duplicate_type": "exact" if sim > 0.98 else "near",
                    }
                )

                if roi_ids[i] not in duplicate_groups:
                    duplicate_groups[roi_ids[i]] = []
                duplicate_groups[roi_ids[i]].append(roi_ids[j])
                is_duplicate.add(roi_ids[j])

    return duplicate_pairs, duplicate_groups, is_duplicate


def run_feature_deduplication(
    sim_matrix,
    roi_ids,
    feature_matrix,
    primary_threshold=0.95,
    analysis_thresholds=None,
    output_dir=None,
):
    """
    Full deduplication pipeline with multi-threshold analysis.
    """
    if analysis_thresholds is None:
        analysis_thresholds = [0.90, 0.93, 0.95, 0.97, 0.99]

    # Multi-threshold analysis
    threshold_analysis = []
    for thresh in analysis_thresholds:
        pairs, groups, dupes = find_feature_duplicates(sim_matrix, roi_ids, thresh)
        threshold_analysis.append(
            {
                "threshold": thresh,
                "duplicate_pairs_found": len(pairs),
                "unique_rois_remaining": len(roi_ids) - len(dupes),
                "rois_removed": len(dupes),
                "removal_percentage": round(len(dupes) / len(roi_ids) * 100, 1),
            }
        )
        print(
            f"Threshold {thresh:.2f}: {len(pairs)} pairs, "
            f"{len(dupes)} removed ({len(roi_ids) - len(dupes)} remain)"
        )

    # Primary deduplication
    pairs, groups, dupes = find_feature_duplicates(
        sim_matrix, roi_ids, primary_threshold
    )

    clean_roi_ids = [r for r in roi_ids if r not in dupes]
    clean_indices = [roi_ids.index(r) for r in clean_roi_ids]
    clean_features = feature_matrix[clean_indices]

    print(f"\n--- Primary Dedup (threshold={primary_threshold}) ---")
    print(f"Before: {len(roi_ids)} ROIs")
    print(f"After:  {len(clean_roi_ids)} ROIs")
    print(f"Removed: {len(dupes)} duplicates")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)

    pd.DataFrame(pairs).to_csv(output_dir / "05_feature_duplicates.csv", index=False)
    pd.DataFrame(threshold_analysis).to_csv(
        output_dir / "05b_threshold_analysis.csv", index=False
    )

    # THIS is what creates the missing file
    # Check that the path points to data/features/ not data/csvs/
    features_dir = output_dir.parent / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        features_dir / "clean_features.npz",
        **{rid: feature_matrix[roi_ids.index(rid)] for rid in clean_roi_ids},
    )
    with open(features_dir / "clean_roi_ids.txt", "w") as f:
        for rid in clean_roi_ids:
            f.write(f"{rid}\n")

    return clean_roi_ids, clean_features, pairs, threshold_analysis


if __name__ == "__main__":
    from src.config import CONFIG
    from src.stage2_deduplication.similarity_matrix import compute_similarity_matrix

    roi_ids, feat_matrix, sim_matrix = compute_similarity_matrix(
        f"{CONFIG['paths']['features']}/roi_features.npz"
    )

    run_feature_deduplication(
        sim_matrix,
        roi_ids,
        feat_matrix,
        CONFIG["deduplication"]["primary_threshold"],
        CONFIG["deduplication"]["analysis_thresholds"],
        CONFIG["paths"]["csvs"],
    )
