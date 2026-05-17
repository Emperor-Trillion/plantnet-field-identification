"""
Step 1.4: Color-based flower segmentation with automated content screening.
Isolates flower regions and saves cropped ROIs.

Screening layers (applied in order):
  Layer 1 — Person overlap: suppress ROIs overlapping YOLO person bounding boxes
  Layer 2 — CLIP screening: reject ROIs CLIP considers non-botanical

NOTE: A size filter was considered but removed. Zoomed-in flower photographs
legitimately produce ROIs covering 50-90% of image area. Size alone cannot
distinguish a macro flower shot from a person's clothing crop — CLIP handles
both cases correctly regardless of ROI size, making a size filter both
unnecessary and harmful to recall on close-up images.

Outputs:
  03_flower_rois.csv      — accepted ROIs only (feeds rest of pipeline)
  03b_screened_rois.csv   — rejected ROIs with rejection layer, reason,
                            CLIP scores, and path to saved crop image
  data/cropped_rois/      — accepted crop images
  data/screened_rois/     — rejected crop images (for audit and paper figures)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils.image_utils import get_image_files
import json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IoU threshold above which an ROI is suppressed for overlapping a person box.
# Deliberately low — even partial overlap with a detected person suppresses the ROI.
PERSON_OVERLAP_IOU_THRESHOLD = 0.15

# CLIP prompts.
# Accept: what a genuine botanical ROI looks like.
# Reject: what should never reach PlantNet classification.
CLIP_ACCEPT_PROMPTS = [
    "a flower",
    "a plant",
    "a botanical specimen",
    "flower petals",
    "vegetation with blooms",
]
CLIP_REJECT_PROMPTS = [
    "a person",
    "a human face",
    "a hand",
    "clothing or fabric",
    "an artificial object",
    "a building",
    "a road or pavement",
    "a car or vehicle",
    "a sign or text",
]


# ---------------------------------------------------------------------------
# Person overlap helpers
# ---------------------------------------------------------------------------


def _iou(box_a, box_b):
    """Intersection-over-union between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _overlapping_person_iou(roi_box, person_boxes):
    """
    Return the highest IoU between roi_box and any person box.
    Returns 0.0 if person_boxes is empty.
    """
    if not person_boxes:
        return 0.0
    return max(_iou(roi_box, pb) for pb in person_boxes)


def _load_person_boxes(yolo_json_path):
    """
    Parse 02_yolo_detections.json.
    Returns { image_filename: [[x1,y1,x2,y2], ...] }
    Returns empty dict if file does not exist.
    """
    yolo_path = Path(yolo_json_path)
    if not yolo_path.exists():
        print(f"   [WARN] YOLO JSON not found: {yolo_path}")
        print(f"   [WARN] Person-box suppression will be skipped.")
        return {}

    with open(yolo_path) as f:
        detections = json.load(f)

    person_map = {}
    for det in detections:
        boxes = det.get("person_boxes", [])
        if boxes:
            person_map[det["image_id"]] = boxes

    total_boxes = sum(len(v) for v in person_map.values())
    print(
        f"   Loaded person boxes: {total_boxes} boxes across "
        f"{len(person_map)} images"
    )
    return person_map


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------


def _load_clip_model():
    """
    Load CLIP ViT-B/32 onto the best available device.
    Returns (model, preprocess) on success, (None, None) on ImportError.
    """
    try:
        import clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        print(f"   CLIP model loaded (ViT-B/32) on {device}")
        return model, preprocess
    except ImportError:
        print("   [WARN] clip package not installed.")
        print("   [WARN] Run: pip install git+https://github.com/openai/CLIP.git")
        return None, None


def _clip_scores(crop_bgr, clip_model, clip_preprocess):
    """
    Score a BGR crop against accept and reject prompt sets.

    Returns (accept_score, reject_score) as plain floats.
    accept_score + reject_score = 1.0 (softmax over all prompts combined).

    A crop is considered non-botanical when reject_score > accept_score.
    This applies regardless of the crop's size relative to the image —
    a black jacket filling 90% of the frame is rejected just as reliably
    as a small clothing fragment in the corner.
    """
    import torch
    import clip
    from PIL import Image

    # Determine which device the model is on and move inputs to match
    device = next(clip_model.parameters()).device

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    image_input = clip_preprocess(pil_img).unsqueeze(0).to(device)

    all_prompts = CLIP_ACCEPT_PROMPTS + CLIP_REJECT_PROMPTS
    text_tokens = clip.tokenize(all_prompts).to(device)

    with torch.no_grad():
        logits, _ = clip_model(image_input, text_tokens)
        probs = logits.softmax(dim=-1)[0]

    accept_score = round(probs[: len(CLIP_ACCEPT_PROMPTS)].sum().item(), 4)
    reject_score = round(probs[len(CLIP_ACCEPT_PROMPTS) :].sum().item(), 4)
    return accept_score, reject_score


# ---------------------------------------------------------------------------
# Core colour segmentation (algorithm unchanged)
# ---------------------------------------------------------------------------


def segment_flower_regions(
    image_path,
    sat_threshold=80,
    green_hue_range=(35, 85),
    min_area_fraction=0.01,
    padding=20,
):
    """
    Segment candidate flower regions using colour analysis.
    Flowers = high saturation AND not green.
    Returns (regions, mask) — raw candidates before any screening.
    No size ceiling is applied here; zoomed-in flower images legitimately
    produce very large ROIs and must not be penalised for it.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return [], None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    sat_mask = s > sat_threshold
    green_mask = (h > green_hue_range[0]) & (h < green_hue_range[1])
    flower_mask = (sat_mask & ~green_mask).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    flower_mask = cv2.morphologyEx(flower_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    flower_mask = cv2.morphologyEx(flower_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        flower_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = img.shape[:2]
    image_area = img_h * img_w
    min_area = image_area * min_area_fraction
    regions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h_box = cv2.boundingRect(contour)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_w, x + w + padding)
            y2 = min(img_h, y + h_box + padding)
            regions.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "area": int(area),
                    "center": [x + w // 2, y + h_box // 2],
                }
            )

    return regions, flower_mask


# ---------------------------------------------------------------------------
# Batch processing with screening and full rejection logging
# ---------------------------------------------------------------------------


def segment_all_images(
    image_dir,
    roi_output_dir,
    csv_output_path,
    config=None,
    yolo_json_path=None,
    use_clip=True,
    screened_output_dir=None,
    screened_csv_path=None,
    observations_csv=None,  # NEW: skips same images excluded in YOLO
):
    """
    Process all images: segment candidates, apply screening layers, save all outputs.

    Accepted ROIs  → roi_output_dir      + csv_output_path
    Rejected ROIs  → screened_output_dir + screened_csv_path
                     Crops are saved to disk so rejections can be audited
                     and used as evidence in the paper.
    """
    image_files = get_image_files(image_dir)
    roi_dir = Path(roi_output_dir)
    roi_dir.mkdir(parents=True, exist_ok=True)

    # Default paths for screened outputs (sibling to cropped_rois/)
    if screened_output_dir is None:
        screened_output_dir = Path(roi_output_dir).parent / "screened_rois"
    if screened_csv_path is None:
        screened_csv_path = Path(csv_output_path).parent / "03b_screened_rois.csv"

    screened_dir = Path(screened_output_dir)
    screened_dir.mkdir(parents=True, exist_ok=True)

    # Config values
    sat_thresh = config["detection"]["flower_saturation_threshold"] if config else 80
    green_range = (
        tuple(config["detection"]["flower_green_hue_range"]) if config else (35, 85)
    )
    min_area = config["detection"]["min_roi_area_fraction"] if config else 0.01
    pad = config["detection"]["roi_padding"] if config else 20

    # Load exclusion list from human observations (same set YOLO skipped)
    excluded = set()
    if observations_csv:
        from src.stage1_detection.yolo_detection import load_excluded_images

        print("\n   Loading image exclusion list for segmentation...")
        excluded = load_excluded_images(observations_csv)

    # Load person boxes for Layer 1
    person_box_map = {}
    if yolo_json_path:
        print("\n   Loading YOLO person boxes for ROI suppression...")
        person_box_map = _load_person_boxes(yolo_json_path)

    # Load CLIP for Layer 2
    clip_model, clip_preprocess = None, None
    if use_clip:
        print("\n   Loading CLIP model for botanical screening...")
        clip_model, clip_preprocess = _load_clip_model()
        if clip_model is None:
            print(
                "   [WARN] CLIP unavailable — non-botanical ROIs will not be "
                "screened. Person-box suppression (Layer 1) remains active."
            )

    accepted_records = []
    rejected_records = []
    roi_counter = 0
    screened_counter = 0

    n_raw = 0
    n_rejected_person = 0
    n_rejected_clip = 0
    n_accepted = 0

    for img_path in tqdm(image_files, desc="Segmenting + screening"):
        img_name = img_path.name

        # Skip images excluded by observation file (same as YOLO)
        if img_name in excluded:
            continue

        regions, _ = segment_flower_regions(
            img_path, sat_thresh, green_range, min_area, pad
        )
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        image_area = img_h * img_w
        person_boxes = person_box_map.get(img_name, [])

        for region in regions:
            n_raw += 1
            x1, y1, x2, y2 = region["bbox"]
            roi_box = [x1, y1, x2, y2]
            roi_area = (x2 - x1) * (y2 - y1)
            area_fraction = roi_area / image_area

            # Fields written for both accepted and rejected records
            base = {
                "source_image": img_name,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "area": region["area"],
                "area_fraction": round(area_fraction, 4),
                "center_x": region["center"][0],
                "center_y": region["center"][1],
            }

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ── Layer 1: Person overlap suppression ───────────────────────
            person_iou = _overlapping_person_iou(roi_box, person_boxes)
            if person_iou > PERSON_OVERLAP_IOU_THRESHOLD:
                n_rejected_person += 1
                sid = f"screened_{screened_counter:04d}"
                spath = screened_dir / f"{sid}.jpg"
                cv2.imwrite(str(spath), crop)
                rejected_records.append(
                    {
                        "roi_id": sid,
                        **base,
                        "rejection_layer": "layer1_person_overlap",
                        "rejection_reason": (
                            f"IoU {person_iou:.3f} with YOLO person box "
                            f"exceeds threshold {PERSON_OVERLAP_IOU_THRESHOLD}"
                        ),
                        "person_iou": round(person_iou, 4),
                        "clip_accept_score": None,
                        "clip_reject_score": None,
                        "crop_path": str(spath),
                    }
                )
                screened_counter += 1
                continue

            # ── Layer 2: CLIP botanical screening ─────────────────────────
            # CLIP is the primary content defence. It correctly rejects:
            #   - Black or dark clothing (no saturation signal for colour filter)
            #   - Colourful clothing that passed colour segmentation
            #   - Any non-plant subject regardless of ROI size
            # A zoomed-in flower filling 80% of the frame passes CLIP just
            # as reliably as a small flower in a wider scene.
            clip_accept, clip_reject = None, None
            if clip_model is not None:
                clip_accept, clip_reject = _clip_scores(
                    crop, clip_model, clip_preprocess
                )
                if clip_reject > clip_accept:
                    n_rejected_clip += 1
                    sid = f"screened_{screened_counter:04d}"
                    spath = screened_dir / f"{sid}.jpg"
                    cv2.imwrite(str(spath), crop)
                    rejected_records.append(
                        {
                            "roi_id": sid,
                            **base,
                            "rejection_layer": "layer2_clip",
                            "rejection_reason": (
                                f"CLIP reject score {clip_reject} > "
                                f"accept score {clip_accept}"
                            ),
                            "person_iou": round(person_iou, 4),
                            "clip_accept_score": clip_accept,
                            "clip_reject_score": clip_reject,
                            "crop_path": str(spath),
                        }
                    )
                    screened_counter += 1
                    continue

            # ── Accepted ──────────────────────────────────────────────────
            roi_id = f"roi_{roi_counter:04d}"
            crop_path = roi_dir / f"{roi_id}.jpg"
            cv2.imwrite(str(crop_path), crop)
            accepted_records.append(
                {
                    "roi_id": roi_id,
                    **base,
                    "crop_path": str(crop_path),
                }
            )
            roi_counter += 1
            n_accepted += 1

    # ── Write CSVs ────────────────────────────────────────────────────────
    df_accepted = pd.DataFrame(accepted_records)
    df_accepted.to_csv(csv_output_path, index=False)

    df_rejected = pd.DataFrame(rejected_records)
    df_rejected.to_csv(screened_csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    n_processed = len(image_files) - len(excluded)
    print(f"\nSegmentation + Screening Summary:")
    print(f"  Images in directory:           {len(image_files)}")
    print(f"  Skipped (excluded list):       {len(excluded)}")
    print(f"  Images processed:              {n_processed}")
    print(f"  Raw candidate ROIs:            {n_raw}")
    print(
        f"  ── Rejected (person overlap):  {n_rejected_person}"
        f"  [IoU > {PERSON_OVERLAP_IOU_THRESHOLD} with YOLO person box]"
    )
    print(
        f"  ── Rejected (CLIP):            {n_rejected_clip}"
        f"  {'[CLIP not installed — layer inactive]' if use_clip and clip_model is None else '[non-botanical content]'}"
    )
    print(f"  Accepted ROIs:                 {n_accepted}")
    print(
        f"  Avg accepted ROIs per image:   " f"{n_accepted / max(n_processed, 1):.1f}"
    )
    print(f"\n  Accepted crops → {roi_output_dir}")
    print(f"  Accepted CSV   → {csv_output_path}")
    print(f"  Rejected crops → {screened_output_dir}")
    print(f"  Rejected CSV   → {screened_csv_path}")

    return df_accepted, df_rejected


if __name__ == "__main__":
    from src.config import CONFIG

    csvs = CONFIG["paths"]["csvs"]
    segment_all_images(
        CONFIG["paths"]["raw_images"],
        CONFIG["paths"]["cropped_rois"],
        f"{csvs}/03_flower_rois.csv",
        CONFIG,
        yolo_json_path=f"{csvs}/02_yolo_detections.json",
        use_clip=True,
    )
