# scripts/cached_api_validation.py
"""
Cached PlantNet API validation across all 4 models.

Strategy:
- Call PlantNet API exactly once per ROI (cached on disk).
- Evaluate each local model's predictions against the same cached API responses.
- Produces one CSV per model + a per-tier summary per model.

Cost: ~421 API calls one-time (well under 500/day). After cache is built,
re-evaluation is free and instant.

Usage:
    python -m scripts.cached_api_validation                 # build + evaluate (default)
    python -m scripts.cached_api_validation --build-only    # populate cache only
    python -m scripts.cached_api_validation --eval-only     # evaluate only, no API
    python -m scripts.cached_api_validation --max-calls 100 # cap API spend this run
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import CONFIG
from src.stage3_classification.api_validation import (
    query_plantnet_api,
    normalize_species,
    normalize_genus,
)
from src.utils.taxonomy_resolver import resolve_many

CACHE_PATH = Path("data/csvs/_plantnet_api_cache.json")

# Adjust these paths if your per-model classification CSVs use different names.
# (See "Before you run it" notes below to verify.)
MODELS = {
    "wide_resnet50_2": "outputs/tables/wide_resnet50_2_06_classification_results.csv",
    "densenet121": "outputs/tables/densenet121_06_classification_results.csv",
    "efficientnet_b0": "outputs/tables/efficientnet_b0_06_classification_results.csv",
    "resnet50": "outputs/tables/resnet50_06_classification_results.csv",
}

# Fallback if no per-model CSVs are found
ACTIVE_MODEL_FALLBACK = ("active_model", "data/csvs/06_classification_results.csv")


# ---------- cache helpers ---------------------------------------------------


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    """Atomic save: write to .tmp then rename, so a crash mid-write can't
    corrupt the cache."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(CACHE_PATH)


# ---------- cache builder ---------------------------------------------------


def build_cache(roi_csv: str, max_calls: Optional[int] = None) -> dict:
    """Call PlantNet API once per ROI, caching results to disk.
    Resumable: skips ROIs already in the cache.
    """
    api_key = CONFIG["api"]["api_key"]
    api_url = CONFIG["api"]["base_url"]
    rate_limit = CONFIG["api"].get("rate_limit_seconds", 0.5)

    df_rois = pd.read_csv(roi_csv)
    if "roi_id" not in df_rois.columns or "crop_path" not in df_rois.columns:
        raise ValueError(f"Expected 'roi_id' and 'crop_path' columns in {roi_csv}")

    cache = load_cache()
    to_process = df_rois[~df_rois["roi_id"].astype(str).isin(cache.keys())]
    print(f"[cache] {len(cache)} ROIs already cached; " f"{len(to_process)} remaining")

    if max_calls is not None:
        to_process = to_process.head(max_calls)
        print(f"[cache] capping this run at {max_calls} new calls")

    if len(to_process) == 0:
        print("[cache] nothing to do")
        return cache

    calls_made = 0
    for _, row in to_process.iterrows():
        roi_id = str(row["roi_id"])
        crop_path = row["crop_path"]

        if not Path(crop_path).exists():
            print(f"  ⚠️  missing file for {roi_id}: {crop_path}")
            continue

        api_preds = query_plantnet_api(crop_path, api_key, api_url)

        if api_preds == "RATE_LIMIT":
            print(
                f"\n  ⚠️  Rate limit hit after {calls_made} new calls. "
                f"Cache has {len(cache)} entries total. "
                f"Re-run tomorrow to resume."
            )
            break

        if api_preds is None:
            # transient error — don't cache, continue
            time.sleep(rate_limit)
            continue

        cache[roi_id] = {
            "timestamp": datetime.utcnow().isoformat(),
            "crop_path": crop_path,
            "predictions": api_preds,
        }
        save_cache(cache)  # crash-safe: save after every successful call
        calls_made += 1

        if calls_made % 25 == 0:
            print(f"  [cache] {calls_made} new calls, " f"{len(cache)} total cached")

        time.sleep(rate_limit)

    print(
        f"[cache] done. {calls_made} new calls this run, " f"{len(cache)} total cached."
    )
    return cache


# ---------- per-model evaluator --------------------------------------------


def evaluate_model_against_cache(
    model_name: str,
    classification_csv: str,
    cache: dict,
    output_dir: str = "data/csvs",
) -> Optional[pd.DataFrame]:
    """Compare a single model's predictions against the cached API responses."""
    if not Path(classification_csv).exists():
        print(f"[eval:{model_name}] missing {classification_csv} — skipping")
        return None

    df = pd.read_csv(classification_csv)
    if "confidence_tier" not in df.columns:
        print(f"[eval:{model_name}] no confidence_tier column — skipping")
        return None

    # Build local species → family map via GBIF (cached, near-instant after first run)
    unique_species = df["pred1_species"].dropna().unique().tolist()
    print(f"[eval:{model_name}] resolving {len(unique_species)} species via GBIF...")
    gbif_map = resolve_many(unique_species)
    local_species_to_family = {
        sp: (info.get("family") or "") for sp, info in gbif_map.items()
    }

    rows = []
    for _, row in df.iterrows():
        roi_id = str(row["roi_id"])
        if roi_id not in cache:
            continue

        api_preds = cache[roi_id]["predictions"]
        if not api_preds:
            continue

        local_sp = row["pred1_species"]
        local_gn = row.get("pred1_genus", "")
        local_sp_norm = normalize_species(local_sp)
        local_gn_norm = normalize_genus(local_gn)
        local_family = (local_species_to_family.get(local_sp, "") or "").lower()

        api_top1 = api_preds[0]
        api_sp_norm = normalize_species(api_top1["species"])
        api_gn_norm = normalize_genus(api_top1["genus"])
        api_family = (api_top1["family"] or "").lower()

        api_top5_species = [normalize_species(p["species"]) for p in api_preds]
        api_top5_genus = [normalize_genus(p["genus"]) for p in api_preds]
        api_top5_family = [(p["family"] or "").lower() for p in api_preds]

        rows.append(
            {
                "roi_id": roi_id,
                "tier": row["confidence_tier"],
                "local_conf": row.get("top1_confidence", row.get("pred1_conf")),
                "local_species": local_sp,
                "local_species_norm": local_sp_norm,
                "local_genus_norm": local_gn_norm,
                "local_family": local_family,
                "api_species": api_top1["species"],
                "api_species_norm": api_sp_norm,
                "api_genus_norm": api_gn_norm,
                "api_family": api_family,
                "api_conf": api_top1["confidence"],
                "species_match_top1": int(
                    local_sp_norm == api_sp_norm and local_sp_norm != ""
                ),
                "species_match_top5": int(
                    local_sp_norm in api_top5_species and local_sp_norm != ""
                ),
                "genus_match_top1": int(
                    local_gn_norm == api_gn_norm and local_gn_norm != ""
                ),
                "genus_match_top5": int(
                    local_gn_norm in api_top5_genus and local_gn_norm != ""
                ),
                "family_match_top1": int(
                    local_family == api_family and local_family != ""
                ),
                "family_match_top5": int(
                    local_family in api_top5_family and local_family != ""
                ),
            }
        )

    if not rows:
        print(f"[eval:{model_name}] no matched ROIs — cache empty?")
        return None

    out = pd.DataFrame(rows)
    out_path = Path(output_dir) / f"07_api_cross_validation_{model_name}.csv"
    out.to_csv(out_path, index=False)
    print(f"[eval:{model_name}] wrote {len(out)} rows → {out_path}")

    # Per-tier summary
    summary = (
        out.groupby("tier")
        .agg(
            n=("roi_id", "count"),
            species_top1=("species_match_top1", "mean"),
            species_top5=("species_match_top5", "mean"),
            genus_top1=("genus_match_top1", "mean"),
            genus_top5=("genus_match_top5", "mean"),
            family_top1=("family_match_top1", "mean"),
            family_top5=("family_match_top5", "mean"),
        )
        .round(3)
        .reset_index()
    )
    summary_path = Path(output_dir) / f"07b_api_agreement_by_tier_{model_name}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[eval:{model_name}] summary → {summary_path}")
    print(summary.to_string(index=False))

    return out


# ---------- CLI -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Populate cache only, skip evaluation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate models against existing cache; no API calls.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Cap new API calls this run (safety).",
    )
    parser.add_argument(
        "--roi-csv",
        default="data/csvs/03_flower_rois.csv",
        help="Path to ROI CSV (must contain roi_id, crop_path).",
    )
    args = parser.parse_args()

    if args.build_only and args.eval_only:
        parser.error("--build-only and --eval-only are mutually exclusive")

    # ----- BUILD -----
    if args.eval_only:
        cache = load_cache()
        print(f"[cache] loaded {len(cache)} entries (eval-only mode)")
    else:
        cache = build_cache(args.roi_csv, max_calls=args.max_calls)

    if args.build_only:
        return

    # ----- EVALUATE -----
    models_to_eval = {name: csv for name, csv in MODELS.items() if Path(csv).exists()}
    if not models_to_eval:
        print("\n[eval] no per-model CSVs found at expected paths:")
        for n, p in MODELS.items():
            print(f"   - {n}: {p}")
        print(
            f"[eval] falling back to active-model CSV: " f"{ACTIVE_MODEL_FALLBACK[1]}"
        )
        models_to_eval = dict([ACTIVE_MODEL_FALLBACK])

    print(
        f"\n[eval] evaluating {len(models_to_eval)} model(s) "
        f"against {len(cache)} cached responses"
    )

    for model_name, csv_path in models_to_eval.items():
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        evaluate_model_against_cache(model_name, csv_path, cache)


if __name__ == "__main__":
    main()
