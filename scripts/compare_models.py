import sys
import shutil
import subprocess
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

models_to_test = [
    ("resnet50", "models/resnet50_weights_best_acc.tar"),
    ("efficientnet_b0", "models/efficientnet_b0_weights_best_acc.tar"),
    ("densenet121", "models/densenet121_weights_best_acc.tar"),
    ("wide_resnet50_2", "models/wide_resnet50_2_weights_best_acc.tar"),
]

# All known model prefixes — used to skip already-tagged files
KNOWN_PREFIXES = tuple(f"{name}_" for name, _ in models_to_test)

FIG_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
CSV_DIR = Path("data/csvs")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def tag_and_copy(src_dir: Path, dst_dir: Path, pattern: str, model_name: str):
    """Copy files matching pattern from src_dir to dst_dir, prefixed with model_name.
    Materializes the file list FIRST so newly-created files aren't re-picked up."""
    # Snapshot the list before any copying happens
    files = sorted(src_dir.glob(pattern))
    # Skip files that are already tagged with any model prefix
    files = [f for f in files if not f.name.startswith(KNOWN_PREFIXES)]
    for f in files:
        dst = dst_dir / f"{model_name}_{f.name}"
        shutil.copy(f, dst)
    return len(files)


for model_name, weight_path in models_to_test:
    if not Path(weight_path).exists():
        print(f"Skipping {model_name}: {weight_path} not found")
        continue

    print(f"\n{'='*60}")
    print(f"  RUNNING: {model_name}")
    print(f"{'='*60}")

    # Update config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["paths"]["model_weights"] = weight_path
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run stages 2-4 (check=True so a failed stage stops this model rather than
    # silently propagating bad state into the next stage)
    try:
        subprocess.run(
            ["python", "scripts/run_full_pipeline.py", "--stage", "2"], check=True
        )
        subprocess.run(
            ["python", "scripts/run_full_pipeline.py", "--stage", "3"],
            check=True,
        )
        subprocess.run(
            ["python", "scripts/run_full_pipeline.py", "--stage", "4"], check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"  ❌ {model_name} failed at: {e.cmd}")
        print(f"     Continuing to next model.")
        continue

    # Copy results with model name prefix — using snapshot lists
    n_csv = tag_and_copy(CSV_DIR, TABLE_DIR, "*.csv", model_name)
    n_fig = tag_and_copy(FIG_DIR, FIG_DIR, "*.png", model_name)
    print(f"  ✅ {model_name}: archived {n_csv} CSVs and {n_fig} figures")

print("\nAll models compared. Results in outputs/tables/ and outputs/figures/")
