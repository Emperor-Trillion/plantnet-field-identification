"""
RUN THIS FIRST.
Inspects your .tar model files and .json mapping files
to verify everything is compatible.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch


def inspect_json(filepath):
    print(f"\n{'='*50}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*50}")

    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"Type: {type(data).__name__}")
    print(f"Number of entries: {len(data)}")

    # Show first 5 entries
    items = list(data.items())[:5]
    print(f"First 5 entries:")
    for k, v in items:
        print(f"  '{k}' → '{v}'")

    # Show last 2 entries
    items_last = list(data.items())[-2:]
    print(f"Last 2 entries:")
    for k, v in items_last:
        print(f"  '{k}' → '{v}'")


def inspect_tar(filepath):
    print(f"\n{'='*50}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*50}")

    checkpoint = torch.load(filepath, map_location="cpu")

    if isinstance(checkpoint, dict):
        print(f"Type: dict")
        print(f"Keys: {list(checkpoint.keys())}")

        for key in checkpoint.keys():
            val = checkpoint[key]

            if isinstance(val, dict):
                print(f"\n  '{key}': dict with {len(val)} entries")
                sample_keys = list(val.keys())[:5]
                for sk in sample_keys:
                    if hasattr(val[sk], "shape"):
                        print(f"    '{sk}': tensor shape {val[sk].shape}")
                    else:
                        print(f"    '{sk}': {type(val[sk]).__name__} = {val[sk]}")

            elif isinstance(val, (int, float, str, bool)):
                print(f"\n  '{key}': {val}")

            elif hasattr(val, "shape"):
                print(f"\n  '{key}': tensor shape {val.shape}")

            else:
                print(f"\n  '{key}': {type(val).__name__}")
    else:
        print(f"Type: {type(checkpoint).__name__}")


def verify_mapping_chain(class_idx_path, species_id_to_name_path):
    print(f"\n{'='*50}")
    print(f"Verifying mapping chain")
    print(f"{'='*50}")

    with open(class_idx_path) as f:
        class_to_species = json.load(f)

    with open(species_id_to_name_path) as f:
        species_to_name = json.load(f)

    # Test the chain
    success = 0
    fail = 0

    for class_idx, species_id in list(class_to_species.items())[:10]:
        species_id_str = str(species_id)
        if species_id_str in species_to_name:
            name = species_to_name[species_id_str]
            print(f"  class {class_idx} → species_id {species_id} → {name}")
            success += 1
        else:
            print(f"  class {class_idx} → species_id {species_id} → NOT FOUND")
            fail += 1

    total_classes = len(class_to_species)
    total_mapped = sum(
        1 for sid in class_to_species.values() if str(sid) in species_to_name
    )

    print(f"\n  Total classes: {total_classes}")
    print(f"  Successfully mapped: {total_mapped}")
    print(f"  Unmapped: {total_classes - total_mapped}")

    if total_mapped == total_classes:
        print(f"  ✅ All classes map to species names correctly")
    else:
        print(f"  ⚠️  Some classes cannot be mapped")


if __name__ == "__main__":
    models_dir = PROJECT_ROOT / "models"

    # Inspect JSON files
    json_files = list(models_dir.glob("*.json"))
    for jf in json_files:
        inspect_json(jf)

    # Inspect TAR files
    tar_files = list(models_dir.glob("*.tar"))
    for tf in tar_files:
        inspect_tar(tf)

    # Verify mapping chain
    class_idx_path = models_dir / "class_idx_to_species_id.json"
    species_name_path = models_dir / "plantnet300K_species_id_2_name.json"

    if class_idx_path.exists() and species_name_path.exists():
        verify_mapping_chain(class_idx_path, species_name_path)

    print(f"\n{'='*50}")
    print("INSPECTION COMPLETE")
    print(f"{'='*50}")
    print("\nCopy the output above and check:")
    print("1. How many classes are in the mapping?")
    print("2. What keys does the .tar checkpoint have?")
    print("3. Does the mapping chain work end-to-end?")
    print("\nThen update config.yaml with the correct:")
    print("  - .tar filename")
    print("  - number of classes (if different from 1081)")
