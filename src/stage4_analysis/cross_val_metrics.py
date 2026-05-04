"""
Step 4.6: KNN cross-validation metrics.

Replaces the circular "model grades its own predictions" evaluation
(replacing the former circular self-evaluation) with a leave-one-out KNN classifier built
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.taxonomy_resolver import resolve_many

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_features(features_npz: str) -> tuple[list[str], np.ndarray]:
    """Load roi_features.npz and return (roi_ids, feature_matrix)."""
    npz = np.load(features_npz, allow_pickle=True)
    roi_ids = list(npz.files)
    matrix = np.stack([npz[k] for k in roi_ids]).astype(np.float32)
    # L2-normalize so dot product == cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return roi_ids, matrix


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap mean and (lo, hi) percentile CI."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float32)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def _normalize_species(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    import re

    cleaned = re.sub(r"\s*\([^)]*\)", " ", name)
    parts = cleaned.strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}".lower()
    return cleaned.strip().lower()


def _normalize_genus(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().split()[0].lower()


# ---------------------------------------------------------------------------
# Core KNN
# ---------------------------------------------------------------------------


def _knn_predict(
    query_idx: int, sims: np.ndarray, labels: np.ndarray, k: int
) -> tuple[str, list[str]]:
    """Return (top1, top5) label predictions for a probe via KNN majority vote.

    sims: 1-D similarity array of length N (probe's similarity to all probes,
          including itself — we mask self-similarity to -inf).
    labels: 1-D label array of length N.
    """
    # Mask self
    sims = sims.copy()
    sims[query_idx] = -np.inf

    # Top-k neighbors by similarity, ordered descending
    if k >= len(sims):
        k = len(sims) - 1
    nn_idx = np.argpartition(-sims, k - 1)[:k]
    nn_idx = nn_idx[np.argsort(-sims[nn_idx])]
    nn_labels = labels[nn_idx]

    # Top-1 = majority vote (ties broken by similarity rank)
    counter = Counter(nn_labels)
    top1 = counter.most_common(1)[0][0]

    # Top-5 = the 5 most-frequent labels in the k-neighborhood,
    # padded with any unique labels we've seen if fewer than 5
    top5_seen = [lbl for lbl, _ in counter.most_common(5)]
    return top1, top5_seen


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_knn_crossval(
    config: dict,
    features_npz: Optional[str] = None,
    classification_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_fig: Optional[str] = None,
    k_values: tuple[int, ...] = (1, 3, 5),
    tier_groups: Optional[dict] = None,
    n_boot: int = 1000,
):
    """Run KNN cross-validation across taxonomic levels and tier groups.

    tier_groups: dict mapping a label string -> list of confidence_tier values.
        Default: {
            "Tier 1 only":    ["TIER_1_HIGH"],
            "Tier 1+2":       ["TIER_1_HIGH", "TIER_2_GOOD"],
            "Tier 1+2+3":     ["TIER_1_HIGH", "TIER_2_GOOD",
                               "TIER_3_MODERATE", "TIER_3_GENUS_OK"],
        }
    """
    paths = config["paths"]
    features_npz = features_npz or f"{paths['features']}/roi_features.npz"
    classification_csv = (
        classification_csv or f"{paths['csvs']}/06_classification_results.csv"
    )
    output_csv = output_csv or f"{paths['csvs']}/14_knn_crossval_metrics.csv"
    output_fig = output_fig or f"{paths['figures']}/knn_crossval.png"

    if tier_groups is None:
        tier_groups = {
            "Tier 1 only": ["TIER_1_HIGH"],
            "Tier 1+2": ["TIER_1_HIGH", "TIER_2_GOOD"],
            "Tier 1+2+3": [
                "TIER_1_HIGH",
                "TIER_2_GOOD",
                "TIER_3_MODERATE",
                "TIER_3_GENUS_OK",
            ],
        }

    print(f"\n  Loading features from: {features_npz}")
    roi_ids, feat_matrix = _load_features(features_npz)
    n_total = len(roi_ids)
    print(f"  Features: {n_total} ROIs × {feat_matrix.shape[1]} dims (L2-normalized)")

    print(f"  Loading classifications from: {classification_csv}")
    df = pd.read_csv(classification_csv)
    df = df.set_index("roi_id")

    # Resolve every unique pred1_species to its GBIF family (cached)
    print("  Resolving species → family via GBIF cache...")
    unique_species = df["pred1_species"].dropna().unique().tolist()
    sp_to_fam = {
        sp: rec.get("family", "") for sp, rec in resolve_many(unique_species).items()
    }

    # Build aligned label arrays (one per ROI in feature order)
    sp_labels = np.array(
        [
            _normalize_species(df.loc[r, "pred1_species"]) if r in df.index else ""
            for r in roi_ids
        ]
    )
    gn_labels = np.array(
        [
            _normalize_genus(df.loc[r, "pred1_genus"]) if r in df.index else ""
            for r in roi_ids
        ]
    )
    fam_labels = np.array(
        [
            (
                sp_to_fam.get(df.loc[r, "pred1_species"], "").lower()
                if r in df.index
                else ""
            )
            for r in roi_ids
        ]
    )
    tiers = np.array(
        [df.loc[r, "confidence_tier"] if r in df.index else "" for r in roi_ids]
    )
    confs = np.array(
        [df.loc[r, "top1_confidence"] if r in df.index else 0.0 for r in roi_ids]
    )

    # Precompute full similarity matrix on the probe pool (we'll subset per group)
    print("  Computing pairwise cosine similarities...")
    sim_full = feat_matrix @ feat_matrix.T  # (N, N)

    # ----- evaluate every (tier_group, k, level) combination ----------
    rows = []
    print("\n  Running KNN cross-validation...")

    for group_label, allowed_tiers in tier_groups.items():
        mask = np.isin(tiers, allowed_tiers)
        n_probe = int(mask.sum())
        if n_probe < 5:
            print(f"    [{group_label}] only {n_probe} ROIs — skipping")
            continue
        probe_idx = np.where(mask)[0]
        print(f"\n    [{group_label}]  n = {n_probe}")

        sub_sims = sim_full[np.ix_(probe_idx, probe_idx)]
        sub_sp = sp_labels[probe_idx]
        sub_gn = gn_labels[probe_idx]
        sub_fam = fam_labels[probe_idx]

        for k in k_values:
            sp_correct = np.zeros(n_probe, dtype=bool)
            gn_correct = np.zeros(n_probe, dtype=bool)
            fam_correct = np.zeros(n_probe, dtype=bool)
            sp_top5 = np.zeros(n_probe, dtype=bool)
            gn_top5 = np.zeros(n_probe, dtype=bool)
            fam_top5 = np.zeros(n_probe, dtype=bool)

            iterator = tqdm(range(n_probe), desc=f"      k={k}", leave=False)
            for i in iterator:
                # Species
                p1, p5 = _knn_predict(i, sub_sims[i], sub_sp, k)
                sp_correct[i] = p1 == sub_sp[i] and sub_sp[i] != ""
                sp_top5[i] = sub_sp[i] in p5 and sub_sp[i] != ""
                # Genus
                p1, p5 = _knn_predict(i, sub_sims[i], sub_gn, k)
                gn_correct[i] = p1 == sub_gn[i] and sub_gn[i] != ""
                gn_top5[i] = sub_gn[i] in p5 and sub_gn[i] != ""
                # Family
                p1, p5 = _knn_predict(i, sub_sims[i], sub_fam, k)
                fam_correct[i] = p1 == sub_fam[i] and sub_fam[i] != ""
                fam_top5[i] = sub_fam[i] in p5 and sub_fam[i] != ""

            for level, top1_arr, top5_arr in [
                ("species", sp_correct, sp_top5),
                ("genus", gn_correct, gn_top5),
                ("family", fam_correct, fam_top5),
            ]:
                m1, l1, h1 = _bootstrap_ci(top1_arr.astype(float), n_boot)
                m5, l5, h5 = _bootstrap_ci(top5_arr.astype(float), n_boot)
                rows.append(
                    {
                        "tier_group": group_label,
                        "n": n_probe,
                        "k": k,
                        "level": level,
                        "top1_acc": m1,
                        "top1_ci_lo": l1,
                        "top1_ci_hi": h1,
                        "top5_acc": m5,
                        "top5_ci_lo": l5,
                        "top5_ci_hi": h5,
                    }
                )

    df_out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    # ----- console report ---------------------------------------------------
    print("\n" + "=" * 78)
    print("KNN CROSS-VALIDATION — leave-one-out, cosine similarity in feature space")
    print("=" * 78)
    print(
        f"{'Group':<14} {'n':>4} {'k':>3} {'Level':<8} "
        f"{'Top-1':>8} {'95% CI':>16}  {'Top-5':>8} {'95% CI':>16}"
    )
    print("-" * 78)
    for _, r in df_out.iterrows():
        print(
            f"{r['tier_group']:<14} {int(r['n']):>4d} {int(r['k']):>3d} "
            f"{r['level']:<8} "
            f"{r['top1_acc']:>7.1%} "
            f"[{r['top1_ci_lo']:.2f}-{r['top1_ci_hi']:.2f}]  "
            f"{r['top5_acc']:>7.1%} "
            f"[{r['top5_ci_lo']:.2f}-{r['top5_ci_hi']:.2f}]"
        )

    # ----- figure ----------------------------------------------------------
    _plot_knn_results(df_out, output_fig)

    print(f"\n  Saved metrics: {output_csv}")
    print(f"  Saved figure:  {output_fig}")
    return df_out


def _plot_knn_results(df: pd.DataFrame, out_path: str):
    """Bar chart: Top-1 accuracy with 95% CI, grouped by level × tier."""
    if df.empty:
        return
    # Use k=5 as the default for the figure
    sub = df[df["k"] == 5].copy()
    if sub.empty:
        sub = df[df["k"] == df["k"].max()].copy()

    levels = ["species", "genus", "family"]
    groups = list(sub["tier_group"].unique())
    x = np.arange(len(levels))
    width = 0.8 / max(1, len(groups))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(groups)))
    for i, grp in enumerate(groups):
        gd = sub[sub["tier_group"] == grp].set_index("level").reindex(levels)
        means = gd["top1_acc"].values
        err_lo = means - gd["top1_ci_lo"].values
        err_hi = gd["top1_ci_hi"].values - means
        ax.bar(
            x + i * width - 0.4 + width / 2,
            means,
            width,
            yerr=[err_lo, err_hi],
            capsize=4,
            color=colors[i],
            label=f"{grp} (n={int(gd['n'].iloc[0])})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([l.title() for l in levels])
    ax.set_ylabel("KNN Top-1 Accuracy (k=5, leave-one-out)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "KNN Cross-Validation Accuracy by Taxonomic Level\n"
        "(error bars = 95% bootstrap CI)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    from src.config import CONFIG

    run_knn_crossval(CONFIG)
