"""
MAIN ENTRY POINT
Runs the entire pipeline from start to finish.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --skip-api
    python scripts/run_full_pipeline.py --stage 2
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, DEVICE, setup_paths, set_seed
from src.utils.image_utils import get_image_files
from src.stage1_detection.merge_detections import merge_all_detections


def print_banner(stage_num, stage_name):
    print(f"\n{'='*60}")
    print(f"  STAGE {stage_num}: {stage_name}")
    print(f"{'='*60}\n")


def run_stage_1():
    """Detection: observation template, YOLO, flower segmentation, merge"""
    print_banner(1, "MULTI-OBJECT DETECTION")

    csvs = CONFIG["paths"]["csvs"]
    obs_path = f"{csvs}/01_human_observations.csv"

    # Step 1.2: Generate observation template
    print("── Step 1.2: Generating observation template ──")
    from src.stage1_detection.observation_template import generate_observation_template

    if not Path(obs_path).exists():
        generate_observation_template(CONFIG["paths"]["raw_images"], obs_path)
        print("\n⚠️  IMPORTANT: Fill in the observation CSV before continuing!")
        print(f"   File: {obs_path}")
        print("   Open it, look at each image, fill in Yes/No columns.")
        print("   Columns to fill: contains_flowers, contains_human")
        print("   (contains_human=yes → image will be excluded from pipeline)")

        response = input("\n   Have you filled in the observations? (y/n): ")
        if response.lower() != "y":
            print("   Please fill in observations first, then re-run.")
            print(
                "   You can restart from Stage 1 with: "
                "python scripts/run_full_pipeline.py --stage 1"
            )
            sys.exit(0)
    else:
        print(f"   Observations file already exists: {obs_path}")

    # Step 1.3: YOLO detection
    # FIX: pass observations_csv so images with humans/no flowers are skipped
    print("\n── Step 1.3: Running YOLO object detection ──")
    from src.stage1_detection.yolo_detection import run_yolo_detection

    run_yolo_detection(
        CONFIG["paths"]["raw_images"],
        f"{csvs}/02_yolo_detections.json",
        CONFIG["detection"]["yolo_model"],
        CONFIG["detection"]["yolo_confidence"],
        observations_csv=obs_path,  # ← NEW
    )

    # Step 1.4: Flower segmentation
    # FIX: pass yolo_json_path so person boxes suppress overlapping ROIs
    print("\n── Step 1.4: Segmenting flower regions ──")
    from src.stage1_detection.flower_segmentation import segment_all_images

    use_clip = CONFIG.get("detection", {}).get("use_clip_screening", False)
    if use_clip:
        print("   CLIP botanical screening: ENABLED")
    else:
        print(
            "   CLIP botanical screening: disabled  "
            "(set detection.use_clip_screening=true in config to enable)"
        )

    segment_all_images(
        CONFIG["paths"]["raw_images"],
        CONFIG["paths"]["cropped_rois"],
        f"{csvs}/03_flower_rois.csv",
        CONFIG,
        yolo_json_path=f"{csvs}/02_yolo_detections.json",  # ← NEW
        use_clip=use_clip,  # ← NEW
        observations_csv=obs_path,  # ← NEW
    )

    # Step 1.5: Merge all detections
    print("\n── Step 1.5: Merging detection results ──")
    from src.stage1_detection.merge_detections import merge_all_detections

    merge_all_detections(
        obs_path,
        f"{csvs}/02_yolo_detections.json",
        f"{csvs}/03_flower_rois.csv",
        f"{csvs}/04_master_image_table.csv",
    )

    print("\n✅ Stage 1 complete.")


def run_stage_2():
    """Feature extraction, similarity computation, deduplication"""
    print_banner(2, "FEATURE-LEVEL DEDUPLICATION")

    csvs = CONFIG["paths"]["csvs"]
    features = CONFIG["paths"]["features"]

    # Step 2.1: Feature extraction
    print("── Step 2.1: Extracting PlantNet features ──")
    from src.stage2_deduplication.feature_extraction import extract_all_features

    extract_all_features(
        f"{csvs}/03_flower_rois.csv",
        CONFIG["paths"]["model_weights"],
        CONFIG["paths"]["class_idx_to_species"],
        f"{features}/roi_features.npz",
        also_extract_full_images=True,
        full_image_dir=CONFIG["paths"]["raw_images"],
        full_output_npz=f"{features}/full_image_features.npz",
        device=str(DEVICE),
    )

    # Step 2.2: Similarity matrix
    print("\n── Step 2.2: Computing similarity matrix ──")
    from src.stage2_deduplication.similarity_matrix import compute_similarity_matrix

    roi_ids, feat_matrix, sim_matrix = compute_similarity_matrix(
        f"{features}/roi_features.npz", features
    )

    # Step 2.3: Feature deduplication
    print("\n── Step 2.3: Running feature-level deduplication ──")
    from src.stage2_deduplication.feature_dedup import run_feature_deduplication

    run_feature_deduplication(
        sim_matrix,
        roi_ids,
        feat_matrix,
        CONFIG["deduplication"]["primary_threshold"],
        CONFIG["deduplication"]["analysis_thresholds"],
        csvs,
    )

    # Step 2.4: Dedup visualization
    print("\n── Step 2.4: Generating deduplication figures ──")
    from src.stage2_deduplication.dedup_visualization import (
        visualize_deduplication_results,
    )

    visualize_deduplication_results(
        sim_matrix,
        roi_ids,
        f"{csvs}/05_feature_duplicates.csv",
        f"{csvs}/05b_threshold_analysis.csv",
        CONFIG["paths"]["cropped_rois"],
        CONFIG["paths"]["figures"],
    )

    print("\n✅ Stage 2 complete.")


def run_stage_3(skip_api=False):
    """Classification, confidence tiering, API validation, label assignment"""
    print_banner(3, "IDENTIFICATION & CLASSIFICATION")

    csvs = CONFIG["paths"]["csvs"]
    features = CONFIG["paths"]["features"]

    # Step 3.1: Zero-shot classification
    print("── Step 3.1: Running PlantNet classification ──")
    from src.stage3_classification.plantnet_classifier import classify_all_rois

    classify_all_rois(
        f"{features}/clean_roi_ids.txt",
        f"{csvs}/03_flower_rois.csv",
        CONFIG["paths"]["model_weights"],
        CONFIG["paths"]["class_idx_to_species"],
        CONFIG["paths"]["species_id_to_name"],
        f"{csvs}/06_classification_results.csv",
        device=str(DEVICE),
    )

    # Step 3.2: Confidence tiering
    print("\n── Step 3.2: Assigning confidence tiers ──")
    from src.stage3_classification.confidence_tiering import assign_confidence_tiers

    assign_confidence_tiers(
        f"{csvs}/06_classification_results.csv",
        f"{csvs}/06_classification_results.csv",
        CONFIG,
    )

    # Step 3.3: API validation
    if not skip_api:
        print("\n── Step 3.3: Cross-validating with PlantNet API ──")
        from src.stage3_classification.api_validation import run_api_validation

        run_api_validation(
            f"{csvs}/06_classification_results.csv",
            f"{csvs}/03_flower_rois.csv",
            f"{csvs}/07_api_cross_validation.csv",
            CONFIG,
            tiers_to_validate=None,  # None = all tiers (the new default)
            max_rois=60,  # No cap — process everything
        )

    else:
        print("\n── Step 3.3: Skipping API validation (--skip-api flag) ──")
        import pandas as pd

        empty_df = pd.DataFrame(
            columns=[
                "roi_id",
                "local_pred_species",
                "local_pred_genus",
                "local_pred_conf",
                "api_pred_species",
                "api_pred_genus",
                "api_pred_family",
                "api_pred_conf",
                "species_match",
                "genus_match",
            ]
        )
        empty_df.to_csv(f"{csvs}/07_api_cross_validation.csv", index=False)

    # Step 3.4: Final label assignment
    print("\n── Step 3.4: Assigning final labels ──")
    from src.stage3_classification.label_assignment import assign_final_labels

    assign_final_labels(
        f"{csvs}/06_classification_results.csv",
        f"{csvs}/07_api_cross_validation.csv",
        f"{csvs}/08_final_labeled_dataset.csv",
    )

    # Step 3.5: Build validated ground truth
    print("\n── Step 3.5: Building validated ground truth ──")
    from src.stage3_classification.build_ground_truth import (
        build_validated_ground_truth,
    )

    build_validated_ground_truth(
        f"{csvs}/06_classification_results.csv",
        f"{csvs}/07_api_cross_validation.csv",
        f"{csvs}/09_validated_ground_truth.csv",
    )

    print("\n✅ Stage 3 complete.")


def run_stage_4():
    """Analysis, visualization, metrics"""
    print_banner(4, "ANALYSIS & VISUALIZATION")

    csvs = CONFIG["paths"]["csvs"]
    features = CONFIG["paths"]["features"]
    figures = CONFIG["paths"]["figures"]

    # Step 4.1 & 4.2: Clustering analysis
    print("── Step 4.1/4.2: Running clustering analysis ──")
    from src.stage4_analysis.clustering_analysis import run_clustering_analysis

    run_clustering_analysis(
        f"{features}/clean_features.npz",
        f"{features}/clean_roi_ids.txt",
        f"{csvs}/08_final_labeled_dataset.csv",
        figures,
        CONFIG,
    )

    # Step 4.3: Grad-CAM
    print("\n── Step 4.3: Generating Grad-CAM visualizations ──")
    from src.stage4_analysis.gradcam_analysis import run_gradcam_analysis

    run_gradcam_analysis(
        f"{csvs}/06_classification_results.csv",
        CONFIG["paths"]["cropped_rois"],
        CONFIG["paths"]["model_weights"],
        CONFIG["paths"]["class_idx_to_species"],  # ← new key
        CONFIG["paths"]["species_id_to_name"],  # ← new key
        figures,
        device=str(DEVICE),
    )

    # Step 4.4: Statistics charts
    print("\n── Step 4.4: Generating statistics and charts ──")
    from src.stage4_analysis.statistics_charts import generate_all_statistics

    generate_all_statistics(
        f"{csvs}/04_master_image_table.csv",
        f"{csvs}/03_flower_rois.csv",
        f"{csvs}/06_classification_results.csv",
        f"{csvs}/08_final_labeled_dataset.csv",
        f"{csvs}/05b_threshold_analysis.csv",
        figures,
        "outputs/tables",
    )

    # Step 4.5: Multi-species analysis
    print("\n── Step 4.5: Multi-species scene analysis ──")
    from src.stage4_analysis.multi_species_analysis import run_multi_species_analysis

    run_multi_species_analysis(
        f"{csvs}/04_master_image_table.csv",
        f"{csvs}/03_flower_rois.csv",
        f"{csvs}/08_final_labeled_dataset.csv",
        CONFIG["paths"]["raw_images"],
        figures,
        "outputs/tables",
    )

    # Step 4.6: KNN cross-validation (defensible accuracy proxy)
    print("\n── Step 4.6: KNN cross-validation metrics ──")
    try:
        from src.stage4_analysis.cross_val_metrics import run_knn_crossval

        run_knn_crossval(CONFIG)
    except Exception as e:
        print(f"   KNN cross-validation note: {e}")

    # Step 4.7: Full image vs ROI comparison
    print("\n── Step 4.7: Full image vs ROI comparison ──")
    try:
        from src.stage4_analysis.comparative_metrics import compare_full_vs_roi

        compare_full_vs_roi(
            f"{csvs}/03_flower_rois.csv",
            f"{csvs}/08_final_labeled_dataset.csv",
            CONFIG["paths"]["raw_images"],
            CONFIG["paths"]["model_weights"],
            CONFIG["paths"]["class_idx_to_species"],
            CONFIG["paths"]["species_id_to_name"],
            csvs,
            device=str(DEVICE),
        )
    except Exception as e:
        print(f"   Comparison note: {e}")

    print("\n✅ Stage 4 complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Plant Identification Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Run specific stage (1-4). Default 0 = run all.",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip PlantNet API validation in Stage 3",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ZERO-EXPERT PLANT IDENTIFICATION PIPELINE")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    set_seed(CONFIG["project"]["seed"])
    setup_paths(CONFIG)

    # Check prerequisites
    weights_path = Path(CONFIG["paths"]["model_weights"])
    if not weights_path.exists():
        print(f"\n❌ ERROR: Model weights not found at {weights_path}")
        print("   Run: bash scripts/download_models.sh")
        sys.exit(1)

    images_dir = Path(CONFIG["paths"]["raw_images"])
    image_count = len(get_image_files(images_dir))
    if image_count == 0:
        print(f"\n❌ ERROR: No images found in {images_dir}")
        print("   Place your images in data/raw_images/")
        sys.exit(1)

    print(f"\n  Found {image_count} images in {images_dir}")
    print(f"  Model weights: {weights_path}")

    start_time = time.time()

    if args.stage == 0:
        run_stage_1()
        run_stage_2()
        run_stage_3(skip_api=args.skip_api)
        run_stage_4()
    elif args.stage == 1:
        run_stage_1()
    elif args.stage == 2:
        run_stage_2()
    elif args.stage == 3:
        run_stage_3(skip_api=args.skip_api)
    elif args.stage == 4:
        run_stage_4()
    else:
        print(f"Invalid stage: {args.stage}. Use 0-4.")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Results in: outputs/figures/ and outputs/tables/")
    print(f"  CSV data in: data/csvs/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
