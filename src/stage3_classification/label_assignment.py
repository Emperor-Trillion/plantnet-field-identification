"""
Step 3.4: Assign final labels using multi-source consensus.
"""

import pandas as pd
from pathlib import Path


def assign_single_label(row, api_lookup):
    """Decision logic for one ROI"""

    roi_id = row["roi_id"]

    # Tier 1: High confidence → trust local model
    if row.get("confidence_tier") == "TIER_1_HIGH":
        return {
            "roi_id": roi_id,
            "source_image": row["source_image"],
            "final_species": row["pred1_species"],
            "final_genus": row["pred1_genus"],
            "label_source": "high_confidence_local",
            "label_confidence": "high",
        }

    # Tier 2: Good confidence + genus consistent
    if row.get("confidence_tier") == "TIER_2_GOOD":
        return {
            "roi_id": roi_id,
            "source_image": row["source_image"],
            "final_species": row["pred1_species"],
            "final_genus": row["pred1_genus"],
            "label_source": "good_confidence_local",
            "label_confidence": "high",
        }

    # Check API agreement
    if roi_id in api_lookup:
        api = api_lookup[roi_id]

        if api.get("species_match", False):
            return {
                "roi_id": roi_id,
                "source_image": row["source_image"],
                "final_species": row["pred1_species"],
                "final_genus": row["pred1_genus"],
                "label_source": "local_api_species_match",
                "label_confidence": "high",
            }

        if api.get("genus_match", False):
            return {
                "roi_id": roi_id,
                "source_image": row["source_image"],
                "final_species": f"{row['pred1_genus']} sp.",
                "final_genus": row["pred1_genus"],
                "label_source": "local_api_genus_match",
                "label_confidence": "medium",
            }

        # API disagrees and has higher confidence
        if api.get("api_pred_conf", 0) > row.get("top1_confidence", 0):
            return {
                "roi_id": roi_id,
                "source_image": row["source_image"],
                "final_species": api["api_pred_species"],
                "final_genus": api["api_pred_genus"],
                "label_source": "api_preferred",
                "label_confidence": "medium",
            }

    # Fallback: use local prediction but mark as uncertain
    return {
        "roi_id": roi_id,
        "source_image": row["source_image"],
        "final_species": f"{row['pred1_genus']} sp. (uncertain)",
        "final_genus": row["pred1_genus"],
        "label_source": "local_uncertain",
        "label_confidence": "low",
    }


def assign_final_labels(classification_csv, api_csv, output_csv):
    """Assign final labels to all ROIs"""

    df_class = pd.read_csv(classification_csv)

    # Load API results if available
    api_lookup = {}
    try:
        df_api = pd.read_csv(api_csv)
        if len(df_api) > 0:
            api_lookup = {row["roi_id"]: row.to_dict() for _, row in df_api.iterrows()}
    except:
        pass

    results = []
    for _, row in df_class.iterrows():
        label = assign_single_label(row, api_lookup)
        results.append(label)

    df_final = pd.DataFrame(results)

    # Merge confidence tier info
    if "confidence_tier" in df_class.columns:
        tier_map = df_class.set_index("roi_id")["confidence_tier"].to_dict()
        df_final["confidence_tier"] = df_final["roi_id"].map(tier_map)

    if "top1_confidence" in df_class.columns:
        conf_map = df_class.set_index("roi_id")["top1_confidence"].to_dict()
        df_final["top1_confidence"] = df_final["roi_id"].map(conf_map)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_csv, index=False)

    print(f"\nFinal Label Assignment:")
    print(f"  Total ROIs labeled: {len(df_final)}")
    print(f"  Unique species: {df_final['final_species'].nunique()}")
    print(f"  Unique genera: {df_final['final_genus'].nunique()}")
    print(f"\n  Label Sources:")
    print(df_final["label_source"].value_counts().to_string())
    print(f"\n  Label Confidence:")
    print(df_final["label_confidence"].value_counts().to_string())

    return df_final


if __name__ == "__main__":
    from src.config import CONFIG

    csvs = CONFIG["paths"]["csvs"]
    assign_final_labels(
        f"{csvs}/06_classification_results.csv",
        f"{csvs}/07_api_cross_validation.csv",
        f"{csvs}/08_final_labeled_dataset.csv",
    )
