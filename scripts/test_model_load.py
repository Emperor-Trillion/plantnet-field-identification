# Save as scripts/test_model_load.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CONFIG
from src.utils.model_utils import load_plantnet_model, load_species_mapping

print("Loading species mapping...")
species_map = load_species_mapping(
    CONFIG["paths"]["class_idx_to_species"], CONFIG["paths"]["species_id_to_name"]
)
print(f"  {len(species_map)} species loaded")
print(f"  Example: class 0 = {species_map[0]}")
print(f"  Example: class 500 = {species_map[500]}")
print(f"  Example: class 1080 = {species_map[1080]}")

print("\nLoading model from .tar...")
model = load_plantnet_model(
    CONFIG["paths"]["model_weights"],
    CONFIG["paths"]["class_idx_to_species"],
    device="cpu",
)
print("  Model loaded successfully")
print(f"  Model type: {type(model).__name__}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

print("\n✅ Everything works. Ready to run full pipeline.")
