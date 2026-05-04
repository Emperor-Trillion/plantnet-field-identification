"""
Build validated ground truth from classification results + API results.
Uses confidence tiers and multi-source agreement.
"""

import pandas as pd
from pathlib import Path


def build_validated_ground_truth(classification_csv, api_csv, output_csv):
    """Create validated ground truth for metric computation"""

    df_class = pd.read_csv(classification_csv)

    # Load API results if available
    try:
        df_api = pd.read_csv(api_csv)
        if len(df_api) == 0:
            df_api = pd.DataFrame()
    except:
        df_api = pd.DataFrame()

    api_lookup = {}
    if len(df_api) > 0:
        api_lookup = {row["roi_id"]: row.to_dict() for _, row in df_api.iterrows()}

    validated = []

    for _, row in df_class.iterrows():
        roi_id = row["roi_id"]
        tier = row.get("confidence_tier", "UNKNOWN")

        # TIER 1: High confidence → trust as ground truth
        if tier == "TIER_1_HIGH":
            validated.append(
                {
                    "roi_id": roi_id,
                    "source_image": row["source_image"],
                    "validated_species": row["pred1_species"],
                    "validated_genus": row["pred1_genus"],
                    "validation_method": "high_confidence_local",
                    "validation_strength": "strong",
                    "include_in_metrics": True,
                }
            )
            continue

        # TIER 2: Good confidence → trust as ground truth
        if tier == "TIER_2_GOOD":
            validated.append(
                {
                    "roi_id": roi_id,
                    "source_image": row["source_image"],
                    "validated_species": row["pred1_species"],
                    "validated_genus": row["pred1_genus"],
                    "validation_method": "good_confidence_local",
                    "validation_strength": "strong",
                    "include_in_metrics": True,
                }
            )
            continue

        # TIER 3 with genus agreement: accept at genus level
        if tier in ["TIER_3_GENUS_OK", "TIER_3_MODERATE"]:
            if row.get("genus_agreement", False):
                validated.append(
                    {
                        "roi_id": roi_id,
                        "source_image": row["source_image"],
                        "validated_species": row["pred1_species"],
                        "validated_genus": row["pred1_genus"],
                        "validation_method": "moderate_with_genus_agreement",
                        "validation_strength": "moderate",
                        "include_in_metrics": True,
                    }
                )
                continue

        # Check API agreement if available
        if roi_id in api_lookup:
            api = api_lookup[roi_id]
            if api.get("species_match", False):
                validated.append(
                    {
                        "roi_id": roi_id,
                        "source_image": row["source_image"],
                        "validated_species": row["pred1_species"],
                        "validated_genus": row["pred1_genus"],
                        "validation_method": "local_api_species_match",
                        "validation_strength": "strong",
                        "include_in_metrics": True,
                    }
                )
                continue
            if api.get("genus_match", False):
                validated.append(
                    {
                        "roi_id": roi_id,
                        "source_image": row["source_image"],
                        "validated_species": row["pred1_species"],
                        "validated_genus": row["pred1_genus"],
                        "validation_method": "local_api_genus_match",
                        "validation_strength": "moderate",
                        "include_in_metrics": True,
                    }
                )
                continue

        # Everything else: not validated enough
        validated.append(
            {
                "roi_id": roi_id,
                "source_image": row["source_image"],
                "validated_species": row["pred1_species"],
                "validated_genus": row["pred1_genus"],
                "validation_method": "unvalidated",
                "validation_strength": "weak",
                "include_in_metrics": False,
            }
        )

    df_validated = pd.DataFrame(validated)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_validated.to_csv(output_csv, index=False)

    included = df_validated[df_validated["include_in_metrics"] == True]
    excluded = df_validated[df_validated["include_in_metrics"] == False]

    print(f"\nGround Truth Validation Summary:")
    print(f"  Total ROIs: {len(df_validated)}")
    print(
        f"  Validated (for metrics): {len(included)} ({len(included)/len(df_validated)*100:.1f}%)"
    )
    print(f"  Excluded: {len(excluded)}")
    print(f"\n  Validation Methods:")
    print(df_validated["validation_method"].value_counts().to_string())
    print(f"\n  Validation Strength:")
    print(df_validated["validation_strength"].value_counts().to_string())

    return df_validated


if __name__ == "__main__":
    from src.config import CONFIG

    csvs = CONFIG["paths"]["csvs"]
    build_validated_ground_truth(
        f"{csvs}/06_classification_results.csv",
        f"{csvs}/07_api_cross_validation.csv",
        f"{csvs}/09_validated_ground_truth.csv",
    )
