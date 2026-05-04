"""
Step 3.3: Cross-validate predictions with the PlantNet API + GBIF taxonomy.

Improvements over the original:
  1. Validates ALL tiers, not just uncertain ones.
  2. Strips taxonomic authors before string matching.
  3. Reports agreement broken down by confidence tier.
  4. NEW: Resolves the local model's family via GBIF, enabling
     family-level agreement comparison (the key metric given that
     genus/species names suffer from taxonomic-drift synonymy).
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.utils.taxonomy_resolver import resolve_many

# --- name normalization -----------------------------------------------------


def normalize_species(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    cleaned = re.sub(r"\s*\([^)]*\)", " ", name)
    parts = cleaned.strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}".lower()
    return cleaned.strip().lower()


def normalize_genus(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().split()[0].lower()


# --- defensive CSV loader ---------------------------------------------------


def _load_with_tiers(classification_csv: str) -> pd.DataFrame:
    df = pd.read_csv(classification_csv)
    if "confidence_tier" in df.columns:
        return df
    csvs_dir = Path(classification_csv).parent
    for cand in sorted(csvs_dir.glob("*.csv")):
        if cand == Path(classification_csv):
            continue
        try:
            head = pd.read_csv(cand, nrows=0)
        except Exception:
            continue
        if {"roi_id", "confidence_tier"}.issubset(head.columns):
            tier_df = pd.read_csv(cand, usecols=["roi_id", "confidence_tier"])
            print(f"  Merged confidence_tier from: {cand.name}")
            return df.merge(tier_df, on="roi_id", how="left")
    raise KeyError(
        f"confidence_tier not found in any CSV under {csvs_dir}. "
        "Run Stage 3 (confidence_tiering step) first."
    )


# --- API call ---------------------------------------------------------------


def query_plantnet_api(image_path: str, api_key: str, api_url: str):
    """Query the PlantNet API. Returns:
    list[dict] : top-5 predictions
    'RATE_LIMIT' : signal to caller that we hit the daily quota
    None : transient error (skip this ROI but continue)
    """
    try:
        with open(image_path, "rb") as img_file:
            response = requests.post(
                api_url,
                files=[("images", img_file)],
                params={
                    "include-related-images": False,
                    "no-reject": False,
                    "lang": "en",
                    "api-key": api_key,
                },
                timeout=30,
            )
    except Exception as e:
        print(f"  Network error on {Path(image_path).name}: {e}")
        return None

    if response.status_code == 429:
        return "RATE_LIMIT"
    if response.status_code in (401, 403):
        # 403 commonly means quota exhausted on the free tier
        if "quota" in response.text.lower() or "limit" in response.text.lower():
            return "RATE_LIMIT"
        print(f"  Auth error ({response.status_code}): {response.text[:200]}")
        return "RATE_LIMIT"  # treat as fatal so we stop wasting attempts
    if response.status_code != 200:
        print(
            f"  HTTP {response.status_code} on {Path(image_path).name}: "
            f"{response.text[:120]}"
        )
        return None

    try:
        data = response.json()
    except Exception as e:
        print(f"  JSON parse error: {e}")
        return None

    results = []
    for r in data.get("results", [])[:5]:
        sp = r.get("species", {})
        results.append(
            {
                "species": sp.get("scientificNameWithoutAuthor", "Unknown"),
                "genus": sp.get("genus", {}).get(
                    "scientificNameWithoutAuthor", "Unknown"
                ),
                "family": sp.get("family", {}).get(
                    "scientificNameWithoutAuthor", "Unknown"
                ),
                "confidence": round(r.get("score", 0), 4),
            }
        )
    return results


# --- main entry -------------------------------------------------------------


def run_api_validation(
    classification_csv,
    roi_csv,
    output_csv,
    config,
    tiers_to_validate=None,
    max_rois=None,
):
    api_key = config["api"]["api_key"]
    api_url = config["api"]["base_url"]
    rate_limit = config["api"]["rate_limit_seconds"]

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("  ⚠️  No API key configured — skipping.")
        return pd.DataFrame()

    df_class = _load_with_tiers(classification_csv)
    df_rois = pd.read_csv(roi_csv)

    if tiers_to_validate is None:
        to_validate = df_class.copy()
    else:
        to_validate = df_class[
            df_class["confidence_tier"].isin(tiers_to_validate)
        ].copy()

    n_tiers_present = max(1, to_validate["confidence_tier"].nunique())

    if max_rois is not None and len(to_validate) > max_rois:
        per_tier = max(1, max_rois // n_tiers_present)
        sampled = []
        for _, group in to_validate.groupby("confidence_tier"):
            sampled.append(group.sample(min(len(group), per_tier), random_state=42))
        to_validate = pd.concat(sampled, ignore_index=True)

    if "confidence_tier" not in to_validate.columns:
        to_validate = to_validate.merge(
            df_class[["roi_id", "confidence_tier"]], on="roi_id", how="left"
        )

    n_total = len(to_validate)
    eta_min = n_total * rate_limit / 60
    print(
        f"Validating {n_total} predictions via API "
        f"(across {to_validate['confidence_tier'].nunique()} tiers)..."
    )
    print(f"Estimated runtime: ~{eta_min:.1f} minutes")

    # Resolve all UNIQUE local species to GBIF families up front (fast, cached)
    print("\n  Resolving local-model species families via GBIF...")
    unique_local_species = to_validate["pred1_species"].dropna().unique().tolist()
    gbif_records = resolve_many(unique_local_species)
    local_species_to_family = {
        sp: rec.get("family", "") for sp, rec in gbif_records.items()
    }

    # --- API loop ---
    api_results = []
    rate_limit_hit = False
    for _, row in tqdm(to_validate.iterrows(), total=n_total, desc="API Validation"):
        roi_info = df_rois[df_rois["roi_id"] == row["roi_id"]]
        if len(roi_info) == 0:
            continue
        roi_path = roi_info.iloc[0]["crop_path"]

        api_preds = query_plantnet_api(roi_path, api_key, api_url)

        if api_preds == "RATE_LIMIT":
            print(
                f"\n  ⚠️  Rate limit reached after {len(api_results)} successful "
                f"calls. Stopping early. Try again in 24 hours."
            )
            rate_limit_hit = True
            break
        if not api_preds:
            time.sleep(rate_limit)
            continue

        # ... (rest of the loop body unchanged)

        local_sp_norm = normalize_species(row["pred1_species"])
        local_gn_norm = normalize_genus(row["pred1_genus"])
        local_family = local_species_to_family.get(row["pred1_species"], "").lower()

        api_sp_norm = normalize_species(api_preds[0]["species"])
        api_gn_norm = normalize_genus(api_preds[0]["genus"])
        api_family = (api_preds[0]["family"] or "").lower()

        api_top5_species = [normalize_species(p["species"]) for p in api_preds]
        api_top5_genus = [normalize_genus(p["genus"]) for p in api_preds]
        api_top5_family = [(p["family"] or "").lower() for p in api_preds]

        api_results.append(
            {
                "roi_id": row["roi_id"],
                "tier": row["confidence_tier"],
                "local_conf": row.get("top1_confidence", row.get("pred1_conf")),
                "local_species": row["pred1_species"],
                "local_species_norm": local_sp_norm,
                "local_genus_norm": local_gn_norm,
                "local_family": local_family,
                "api_species": api_preds[0]["species"],
                "api_species_norm": api_sp_norm,
                "api_genus_norm": api_gn_norm,
                "api_family": api_family,
                "api_conf": api_preds[0]["confidence"],
                "species_match": local_sp_norm == api_sp_norm,
                "genus_match": local_gn_norm == api_gn_norm,
                "family_match": bool(local_family)
                and bool(api_family)
                and local_family == api_family,
                "species_in_api_top5": local_sp_norm in api_top5_species,
                "genus_in_api_top5": local_gn_norm in api_top5_genus,
                "family_in_api_top5": bool(local_family)
                and local_family in api_top5_family,
            }
        )
        time.sleep(rate_limit)

    df_api = pd.DataFrame(api_results)
    if df_api.empty:
        return df_api

    df_api.to_csv(output_csv, index=False)

    # --- reporting ---
    print("\n" + "=" * 76)
    print("API VALIDATION — OVERALL")
    print("=" * 76)
    print(f"  ROIs validated         : {len(df_api)}")
    print(f"  Top-1 species match    : {df_api['species_match'].mean():.1%}")
    print(f"  Top-1 genus   match    : {df_api['genus_match'].mean():.1%}")
    print(f"  Top-1 family  match    : {df_api['family_match'].mean():.1%}  ⭐")
    print(f"  Species in API top-5   : {df_api['species_in_api_top5'].mean():.1%}")
    print(f"  Genus   in API top-5   : {df_api['genus_in_api_top5'].mean():.1%}")
    print(f"  Family  in API top-5   : {df_api['family_in_api_top5'].mean():.1%}  ⭐")

    print("\n" + "=" * 76)
    print("API AGREEMENT BY CONFIDENCE TIER")
    print("=" * 76)
    print(
        f"{'Tier':<22} {'n':>4} {'sp@1':>7} {'gn@1':>7} {'fa@1':>7} "
        f"{'sp@5':>7} {'gn@5':>7} {'fa@5':>7}"
    )
    print("-" * 76)
    tier_order = [
        "TIER_1_HIGH",
        "TIER_2_GOOD",
        "TIER_3_MODERATE",
        "TIER_3_GENUS_OK",
        "TIER_4_LOW",
        "TIER_5_UNRELIABLE",
    ]
    summary_rows = []
    for tier in tier_order:
        sub = df_api[df_api["tier"] == tier]
        if len(sub) == 0:
            continue
        sp1 = sub["species_match"].mean()
        gn1 = sub["genus_match"].mean()
        fa1 = sub["family_match"].mean()
        sp5 = sub["species_in_api_top5"].mean()
        gn5 = sub["genus_in_api_top5"].mean()
        fa5 = sub["family_in_api_top5"].mean()
        print(
            f"{tier:<22} {len(sub):>4d} "
            f"{sp1:>6.1%} {gn1:>6.1%} {fa1:>6.1%} "
            f"{sp5:>6.1%} {gn5:>6.1%} {fa5:>6.1%}"
        )
        summary_rows.append(
            {
                "tier": tier,
                "n": len(sub),
                "species_match": sp1,
                "genus_match": gn1,
                "family_match": fa1,
                "species_in_top5": sp5,
                "genus_in_top5": gn5,
                "family_in_top5": fa5,
            }
        )

    summary_path = str(Path(output_csv).with_name("07b_api_agreement_by_tier.csv"))
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n  Per-tier summary saved to: {summary_path}")
    print(f"  Full validation saved to:  {output_csv}")

    return df_api


if __name__ == "__main__":
    from src.config import CONFIG

    run_api_validation(
        f"{CONFIG['paths']['csvs']}/06_classification_results.csv",
        f"{CONFIG['paths']['csvs']}/03_flower_rois.csv",
        f"{CONFIG['paths']['csvs']}/07_api_cross_validation.csv",
        CONFIG,
    )
