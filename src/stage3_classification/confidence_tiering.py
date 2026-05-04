"""
Step 3.2: Assign confidence tiers to predictions.
"""

import pandas as pd


def assign_confidence_tier(row, config=None):
    """Assign a confidence tier based on prediction scores"""
    conf = row["top1_confidence"]
    gap = row["top1_top2_gap"]
    genus_agree = row["genus_agreement"]

    if config:
        t = config["classification"]["confidence_tiers"]
        t1_conf, t1_gap = t["tier1_min_conf"], t["tier1_min_gap"]
        t2_conf, t2_gap = t["tier2_min_conf"], t["tier2_min_gap"]
        t3_conf = t["tier3_min_conf"]
        t4_conf = t["tier4_min_conf"]
    else:
        t1_conf, t1_gap = 0.80, 0.40
        t2_conf, t2_gap = 0.60, 0.20
        t3_conf = 0.40
        t4_conf = 0.20

    if conf >= t1_conf and gap >= t1_gap:
        return "TIER_1_HIGH"
    elif conf >= t2_conf and gap >= t2_gap:
        return "TIER_2_GOOD"
    elif conf >= t3_conf:
        if genus_agree:
            return "TIER_3_GENUS_OK"
        else:
            return "TIER_3_MODERATE"
    elif conf >= t4_conf:
        return "TIER_4_LOW"
    else:
        return "TIER_5_UNRELIABLE"


def assign_confidence_tiers(classification_csv, output_csv, config=None):
    """Add confidence tiers to classification results"""
    df = pd.read_csv(classification_csv)

    df["confidence_tier"] = df.apply(
        lambda row: assign_confidence_tier(row, config), axis=1
    )

    df.to_csv(output_csv, index=False)

    print("\nConfidence Tier Distribution:")
    print(df["confidence_tier"].value_counts().to_string())

    usable = df[df["confidence_tier"].isin(["TIER_1_HIGH", "TIER_2_GOOD"])]
    print(
        f"\nUsable labels (Tier 1+2): {len(usable)} / {len(df)} "
        f"({len(usable)/len(df)*100:.1f}%)"
    )

    return df


if __name__ == "__main__":
    from src.config import CONFIG

    assign_confidence_tiers(
        f"{CONFIG['paths']['csvs']}/06_classification_results.csv",
        f"{CONFIG['paths']['csvs']}/06_classification_results.csv",  # overwrite
        CONFIG,
    )
