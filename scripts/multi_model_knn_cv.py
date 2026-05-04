"""
Fast-path multi-model KNN cross-validation.

Reuses the proven Stage-2 entry point (run_full_pipeline.py --stage 2)
to regenerate features per model, then runs KNN-CV on top.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS = [
    ("resnet50", "models/resnet50_weights_best_acc.tar"),
    ("efficientnet_b0", "models/efficientnet_b0_weights_best_acc.tar"),
    ("densenet121", "models/densenet121_weights_best_acc.tar"),
    ("wide_resnet50_2", "models/wide_resnet50_2_weights_best_acc.tar"),
]
CONFIG_PATH = Path("config.yaml")


def update_model_in_config(weight_path: str):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["model_weights"] = weight_path
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def main():
    archive_dir = Path("outputs/tables")
    archive_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("outputs/figures")

    original_cfg = CONFIG_PATH.read_text()
    all_results = []

    try:
        for model_name, weight_path in MODELS:
            if not Path(weight_path).exists():
                print(f"\n⚠️  Skipping {model_name}: weights not found")
                continue

            print(f"\n{'='*70}")
            print(f"  MODEL: {model_name}")
            print(f"{'='*70}")

            update_model_in_config(weight_path)

            # 1. Run Stage 2 (regenerates features for this model)
            print(f"\n  Running Stage 2 (feature extraction + dedup)...")
            r = subprocess.run(
                ["python", "scripts/run_full_pipeline.py", "--stage", "2"],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(f"  ❌ Stage 2 failed:\n{r.stderr[-400:]}")
                continue

            # 2. We must also re-run Stage 3 because Stage 2 only updates
            #    features; KNN-CV reads classification labels from Stage 3
            #    and those depend on the model too.
            print(f"  Running Stage 3 (classification, --skip-api)...")
            r = subprocess.run(
                [
                    "python",
                    "scripts/run_full_pipeline.py",
                    "--stage",
                    "3",
                    "--skip-api",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(f"  ❌ Stage 3 failed:\n{r.stderr[-400:]}")
                continue

            # 3. Run KNN cross-validation
            print(f"  Running KNN-CV...")
            r = subprocess.run(
                [
                    "python",
                    "-c",
                    "from src.config import CONFIG; "
                    "from src.stage4_analysis.cross_val_metrics import run_knn_crossval; "
                    "run_knn_crossval(CONFIG)",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(f"  ❌ KNN-CV failed:\n{r.stderr[-400:]}")
                continue
            # Echo the KNN-CV output
            print(r.stdout[-2500:])

            # 4. Archive metrics CSV with model prefix
            src_csv = Path("data/csvs/14_knn_crossval_metrics.csv")
            src_fig = fig_dir / "knn_crossval.png"
            if src_csv.exists():
                dst_csv = archive_dir / f"{model_name}_14_knn_crossval_metrics.csv"
                shutil.copy(src_csv, dst_csv)

                import pandas as pd

                d = pd.read_csv(src_csv)
                d["model"] = model_name
                all_results.append(d)
                print(f"  ✅ Archived: {dst_csv.name}")
            if src_fig.exists():
                shutil.copy(src_fig, fig_dir / f"{model_name}_knn_crossval.png")

        # 5. Combined cross-model comparison
        if all_results:
            import pandas as pd

            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(archive_dir / "combined_knn_crossval.csv", index=False)

            print(f"\n{'='*78}")
            print("  CROSS-MODEL KNN-CV — Tier 1 only, k=1")
            print(f"{'='*78}")
            head = combined[
                (combined["k"] == 1) & (combined["tier_group"] == "Tier 1 only")
            ]
            pivot = head.pivot_table(
                index="model", columns="level", values="top1_acc"
            ).reindex(columns=["species", "genus", "family"])
            print((pivot * 100).round(1).to_string())
            print(f"\n  Combined: {archive_dir / 'combined_knn_crossval.csv'}")

    finally:
        CONFIG_PATH.write_text(original_cfg)
        print("\n  Original config.yaml restored.")


if __name__ == "__main__":
    main()
