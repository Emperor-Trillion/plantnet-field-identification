"""
Common image loading and processing utilities.
Used across all stages.
"""

from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms


def get_image_files(image_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """Get sorted list of all image files in directory"""
    image_dir = Path(image_dir)
    files = []
    for ext in extensions:
        files.extend(image_dir.glob(f"*{ext}"))
        files.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def load_image_pil(image_path):
    """Load image as PIL RGB"""
    return Image.open(image_path).convert("RGB")


def load_image_cv2(image_path):
    """Load image as OpenCV BGR numpy array"""
    return cv2.imread(str(image_path))


def cv2_to_rgb(image):
    """Convert OpenCV BGR to RGB"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_preprocess_transform(
    image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    """Standard preprocessing for PlantNet inference"""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def crop_region(image, bbox, padding=20):
    """Crop a bounding box region from image with padding"""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return image[int(y1) : int(y2), int(x1) : int(x2)]
