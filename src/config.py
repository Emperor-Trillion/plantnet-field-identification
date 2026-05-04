import yaml
from pathlib import Path
import os
import torch
import random
import numpy as np


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_paths(config):
    """Create all required directories"""
    path_keys = ["raw_images", "cropped_rois", "features", "csvs"]
    for key in path_keys:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)

    Path(config["paths"]["figures"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["tables"]).mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Load config on import
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG = load_config(PROJECT_ROOT / "config.yaml")
DEVICE = get_device()
