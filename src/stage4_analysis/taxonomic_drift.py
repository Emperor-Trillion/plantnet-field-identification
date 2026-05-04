"""
Step 4.x: Taxonomic-drift analysis.

For every unique species predicted by the local PlantNet-300K classifier,
query GBIF to determine:
  - Is the name still ACCEPTED, a SYNONYM, or DOUBTFUL in 2024+ taxonomy?
  - If a SYNONYM, what is the modern accepted name?
  - What family does the name resolve to?

Output: a CSV and a summary figure quantifying how much of the local
model's vocabulary is taxonomically obsolete relative to GBIF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.taxonomy_resolver import resolve_many


def run_taxonomic_drift_analysis(
    classification_csv: str,
    output_dir: str = "outputs/tables",
    figures_dir: str = "outputs/figures",
):
    """Quantify taxonomic drift between local model vocab and GBIF backbone."""
    df = pd.read_csv(classification_csv)
    if "pred1_species" not in df.columns:
        raise KeyError("classification CSV missing 'pred1_species'")

    # Each unique species the local model ever produced as top-1
    unique_species = df["pred1_species"].dropna().unique().tolist()
    print(
        f"\n  Analyzing taxonomic drift for "
        f"{len(unique_species)} unique local-model species..."
    )

    records = resolve_many(unique_species)

    # Build the drift table
    rows = []
    for sp_name, rec in records.items():
        rows.append(
            {
                "local_species": sp_name,
                "cleaned": rec.get("cleaned", ""),
                "gbif_match_type": rec.get("match_type", "NONE"),
                "gbif_status": rec.get("status", ""),
                "is_synonym": rec.get("synonym", False),
                "gbif_accepted": rec.get("accepted_name", ""),
                "gbif_genus": rec.get("genus", ""),
                "gbif_family": rec.get("family", ""),
                "gbif_confidence": rec.get("confidence", 0),
                "gbif_rank": rec.get("rank", ""),
            }
        )
    drift_df = pd.DataFrame(rows)

    # Frequency: how often did the local model use each species name?
    freq = (
        df["pred1_species"]
        .value_counts()
        .rename_axis("local_species")
        .reset_index(name="local_usage_count")
    )
    drift_df = drift_df.merge(freq, on="local_species", how="left").fillna(
        {"local_usage_count": 0}
    )
    drift_df["local_usage_count"] = drift_df["local_usage_count"].astype(int)

    # Did the binomial change between local prediction and GBIF accepted?
    drift_df["name_changed"] = drift_df.apply(
        lambda r: bool(r["gbif_accepted"])
        and r["gbif_accepted"].lower() != r["cleaned"].lower(),
        axis=1,
    )

    # Save full table
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "taxonomic_drift.csv"
    drift_df.sort_values("local_usage_count", ascending=False).to_csv(
        csv_path, index=False
    )

    # ------- summary statistics -------
    total = len(drift_df)
    matched = (drift_df["gbif_match_type"] != "NONE").sum()
    accepted = (drift_df["gbif_status"] == "ACCEPTED").sum()
    synonyms = (drift_df["is_synonym"]).sum()
    renamed = drift_df["name_changed"].sum()
    fuzzy = (drift_df["gbif_match_type"] == "FUZZY").sum()
    higher_rank = (drift_df["gbif_match_type"] == "HIGHERRANK").sum()
    no_match = (drift_df["gbif_match_type"] == "NONE").sum()

    # Weight by usage too (drift's downstream impact on actual predictions)
    total_preds = drift_df["local_usage_count"].sum()
    syn_preds = drift_df.loc[drift_df["is_synonym"], "local_usage_count"].sum()
    renamed_preds = drift_df.loc[drift_df["name_changed"], "local_usage_count"].sum()

    print("\n" + "=" * 64)
    print("TAXONOMIC DRIFT — LOCAL MODEL vs GBIF BACKBONE")
    print("=" * 64)
    print(f"  Unique species in local vocab : {total}")
    print(f"  Matched in GBIF              : {matched} ({matched/total:.1%})")
    print(f"    – ACCEPTED status          : {accepted}")
    print(f"    – Flagged as SYNONYM       : {synonyms}")
    print(f"    – Renamed to new binomial  : {renamed}")
    print(f"    – Match was FUZZY (spelling): {fuzzy}")
    print(f"    – Match only at higher rank: {higher_rank}")
    print(f"    – No match in GBIF         : {no_match}")
    print()
    print(f"  Predictions affected by drift (weighted by usage):")
    print(
        f"    – Synonym name predictions : {syn_preds}/{total_preds} "
        f"({syn_preds/max(total_preds,1):.1%})"
    )
    print(
        f"    – Renamed name predictions : {renamed_preds}/{total_preds} "
        f"({renamed_preds/max(total_preds,1):.1%})"
    )

    # ------- figure: drift composition -------
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: pie of match types
    counts = drift_df["gbif_match_type"].value_counts()
    axes[0].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=plt.cm.Set3.colors,
    )
    axes[0].set_title(
        f"GBIF Match Type Distribution\n" f"({total} unique local-model species)"
    )

    # Right: bar of taxonomic status (weighted by predictions made)
    status_counts = (
        drift_df.groupby("gbif_status")["local_usage_count"]
        .sum()
        .sort_values(ascending=True)
    )
    axes[1].barh(status_counts.index, status_counts.values, color="steelblue")
    axes[1].set_xlabel("Total predictions made (weighted by usage)")
    axes[1].set_title("GBIF Taxonomic Status of Local Predictions")
    for i, v in enumerate(status_counts.values):
        axes[1].text(v, i, f" {int(v)}", va="center")

    plt.tight_layout()
    fig_path = fig_dir / "taxonomic_drift.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ------- top-N drift table for the paper -------
    top_drifts = (
        drift_df[drift_df["name_changed"]]
        .sort_values("local_usage_count", ascending=False)
        .head(15)[
            [
                "local_species",
                "gbif_accepted",
                "gbif_status",
                "gbif_family",
                "local_usage_count",
            ]
        ]
    )
    print("\n  Top 15 most-impactful renames (local → GBIF accepted):")
    if len(top_drifts) == 0:
        print("    (no renamed species detected)")
    else:
        for _, r in top_drifts.iterrows():
            print(
                f"    {r['local_species']:<45s} → "
                f"{r['gbif_accepted']:<35s} "
                f"[{r['gbif_status']}, {int(r['local_usage_count'])} uses]"
            )

    print(f"\n  Saved drift table: {csv_path}")
    print(f"  Saved drift figure: {fig_path}")
    return drift_df
