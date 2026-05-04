"""
GBIF taxonomy resolver.

Resolves any plant scientific name to its full backbone taxonomy
(family, genus, accepted name, synonym status). Used to:
  - Compute family-level agreement between local model and PlantNet API.
  - Detect taxonomic drift (synonyms, renames, splits) between the
    PlantNet-300K training snapshot and the current GBIF backbone.

The GBIF /species/match endpoint is free and has no documented rate
limit for taxonomy lookups. We still cache results to disk so repeated
runs are fast.
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import requests

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
DEFAULT_CACHE = Path("data/csvs/_gbif_taxonomy_cache.json")


# ---------------------------------------------------------------------------
# Name cleaning
# ---------------------------------------------------------------------------


def clean_scientific_name(name: str) -> str:
    """Strip taxonomic authors / parenthetical content from a name.

    Examples:
        'Anemone alpina L.'                    -> 'Anemone alpina'
        "Pelargonium zonale (L.) L'Hér."       -> 'Pelargonium zonale'
        'Alocasia zebrina Schott ex Van Houtte' -> 'Alocasia zebrina'
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    # Remove parenthetical content (authors)
    cleaned = re.sub(r"\s*\([^)]*\)", " ", name)
    parts = cleaned.strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Disk-backed cache
# ---------------------------------------------------------------------------


class TaxonomyCache:
    """Persistent cache of GBIF lookups, keyed by cleaned scientific name."""

    def __init__(self, path: Path = DEFAULT_CACHE):
        self.path = Path(path)
        self.data: dict = {}
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, name: str) -> Optional[dict]:
        return self.data.get(name)

    def set(self, name: str, value: dict):
        self.data[name] = value

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# GBIF lookup
# ---------------------------------------------------------------------------


def resolve_one(name: str, cache: TaxonomyCache, polite_delay: float = 0.05) -> dict:
    """Resolve one species name to GBIF taxonomy. Uses cache if present.

    Returns a dict with keys:
        query, canonical, accepted_name, family, genus, kingdom, status,
        match_type, confidence, synonym, usage_key
    or all-empty fields if no match.
    """
    cleaned = clean_scientific_name(name)
    if not cleaned:
        return _empty_record(name)

    cached = cache.get(cleaned)
    if cached is not None:
        return cached

    try:
        r = requests.get(
            GBIF_MATCH_URL,
            params={"name": cleaned, "kingdom": "Plantae", "verbose": "false"},
            timeout=15,
        )
        if r.status_code != 200:
            rec = _empty_record(name, error=f"HTTP {r.status_code}")
        else:
            data = r.json()
            rec = {
                "query": name,
                "cleaned": cleaned,
                "canonical": data.get("canonicalName", ""),
                "accepted_name": data.get("species", "")
                or data.get("canonicalName", ""),
                "family": (data.get("family") or "").lower(),
                "genus": (data.get("genus") or "").lower(),
                "kingdom": (data.get("kingdom") or "").lower(),
                "status": data.get("status", ""),
                "match_type": data.get("matchType", "NONE"),
                "confidence": data.get("confidence", 0),
                "synonym": bool(data.get("synonym", False)),
                "usage_key": data.get("usageKey", None),
                "rank": data.get("rank", ""),
            }
    except Exception as e:
        rec = _empty_record(name, error=str(e))

    cache.set(cleaned, rec)
    if polite_delay:
        time.sleep(polite_delay)
    return rec


def _empty_record(name: str, error: str = "") -> dict:
    return {
        "query": name,
        "cleaned": clean_scientific_name(name),
        "canonical": "",
        "accepted_name": "",
        "family": "",
        "genus": "",
        "kingdom": "",
        "status": "",
        "match_type": "NONE",
        "confidence": 0,
        "synonym": False,
        "usage_key": None,
        "rank": "",
        "error": error,
    }


def resolve_many(
    names: list[str], cache_path: Path = DEFAULT_CACHE, verbose: bool = True
) -> dict[str, dict]:
    """Resolve a list of names. Returns dict {original_name: record}.

    De-duplicates internally so each unique cleaned name only hits the API
    once across the entire run.
    """
    cache = TaxonomyCache(cache_path)
    out = {}
    unique_cleaned = {}
    for n in names:
        c = clean_scientific_name(n)
        if c and c not in unique_cleaned:
            unique_cleaned[c] = n  # remember one original spelling per cleaned

    if verbose:
        n_new = sum(1 for c in unique_cleaned if cache.get(c) is None)
        print(
            f"  GBIF resolution: {len(unique_cleaned)} unique names "
            f"({n_new} new lookups, {len(unique_cleaned) - n_new} cached)"
        )

    # Iterate progress-style if tqdm is available; otherwise plain loop
    try:
        from tqdm import tqdm

        iterator = tqdm(
            unique_cleaned.items(), total=len(unique_cleaned), desc="Resolving via GBIF"
        )
    except ImportError:
        iterator = unique_cleaned.items()

    for cleaned, original in iterator:
        if cache.get(cleaned) is None:
            resolve_one(original, cache)

    cache.save()

    # Build output keyed by the user's original names (mapping each back to
    # its cleaned-name cache entry)
    for n in names:
        c = clean_scientific_name(n)
        out[n] = cache.get(c) if c else _empty_record(n)
    return out
