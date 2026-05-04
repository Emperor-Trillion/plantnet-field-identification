"""
Model loading and management utilities.
Handles all PlantNet-300K model architectures.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import json


# ──────────────────────────────────────────
# SPECIES MAPPING
# ──────────────────────────────────────────


def load_species_mapping(class_idx_to_species_path, species_id_to_name_path):
    with open(class_idx_to_species_path, "r") as f:
        class_idx_to_species_id = json.load(f)

    with open(species_id_to_name_path, "r") as f:
        species_id_to_name = json.load(f)

    species_map = {}
    unmapped = 0

    for class_idx_str, species_id in class_idx_to_species_id.items():
        class_idx = int(class_idx_str)
        species_id_str = str(species_id)

        if species_id_str in species_id_to_name:
            species_map[class_idx] = species_id_to_name[species_id_str]
        else:
            species_map[class_idx] = f"unknown_species_{species_id}"
            unmapped += 1

    print(f"Species mapping loaded: {len(species_map)} classes")
    if unmapped > 0:
        print(f"  Warning: {unmapped} classes could not be mapped to names")

    return species_map


def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def get_num_classes(class_idx_to_species_path):
    with open(class_idx_to_species_path, "r") as f:
        mapping = json.load(f)
    return len(mapping)


# ──────────────────────────────────────────
# MODEL BUILDERS — ALL ARCHITECTURES
# ──────────────────────────────────────────


def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet34(num_classes):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_wide_resnet101(num_classes):
    model = models.wide_resnet101_2(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_wide_resnet50(num_classes):
    model = models.wide_resnet50_2(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes):
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_densenet161(num_classes):
    model = models.densenet161(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def build_efficientnet_b0_timm(num_classes):
    import timm

    model = timm.create_model("efficientnet_b0", weights=None, num_classes=num_classes)
    return model


def build_densenet121_timm(num_classes):
    import timm

    model = timm.create_model("densenet121", weights=None, num_classes=num_classes)
    return model


def build_wide_resnet50_2_timm(num_classes):
    import timm

    model = timm.create_model("wide_resnet50_2", weights=None, num_classes=num_classes)
    return model


# ──────────────────────────────────────────
# MAIN MODEL LOADER
# ──────────────────────────────────────────


def load_plantnet_model(weights_tar_path, class_idx_to_species_path, device="cpu"):
    num_classes = get_num_classes(class_idx_to_species_path)

    checkpoint = torch.load(weights_tar_path, map_location=device)

    # Extract state dict and arch field
    arch = ""
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        arch = checkpoint.get("arch", "")
        if arch:
            print(f"Architecture field: '{arch}'")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean state dict
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("module."):
            new_key = new_key[7:]
        cleaned_state_dict[new_key] = v

    # Detect architecture
    first_key = list(cleaned_state_dict.keys())[0]
    num_keys = len(cleaned_state_dict)
    arch_lower = arch.lower() if isinstance(arch, str) else ""

    print(f"First weight key: '{first_key}'")
    print(f"Total weight keys: {num_keys}")

    # timm EfficientNet
    if first_key == "conv_stem.weight":
        print("  Detected: EfficientNet-B0 (timm)")
        model = build_efficientnet_b0_timm(num_classes)

    # timm DenseNet
    elif first_key == "features.conv0.weight":
        print("  Detected: DenseNet-121 (timm)")
        model = build_densenet121_timm(num_classes)

    # conv1.weight — could be ResNet50 or Wide ResNet50
    elif first_key == "conv1.weight":
        print("  First key is conv1.weight — trying models...")

        # Try Wide ResNet first
        try:
            print("  Trying: Wide ResNet-50-2 (timm)")
            test_model = build_wide_resnet50_2_timm(num_classes)
            test_model.load_state_dict(cleaned_state_dict)
            print("  Weights loaded successfully")
            test_model = test_model.to(device)
            test_model.eval()
            return test_model
        except Exception as e:
            print(f"  Wide ResNet didn't match: {str(e)[:80]}")

        # Try regular ResNet50
        try:
            print("  Trying: ResNet-50 (torchvision)")
            test_model = build_resnet50(num_classes)
            test_model.load_state_dict(cleaned_state_dict)
            print("  Weights loaded successfully")
            test_model = test_model.to(device)
            test_model.eval()
            return test_model
        except Exception as e:
            print(f"  ResNet-50 didn't match: {str(e)[:80]}")

        print("  Defaulting to ResNet-50 with flexible loading")
        model = build_resnet50(num_classes)

    elif "resnet18" in arch_lower:
        print("  Detected: ResNet-18")
        model = build_resnet18(num_classes)

    elif "resnet34" in arch_lower:
        print("  Detected: ResNet-34")
        model = build_resnet34(num_classes)

    else:
        print("  Detected: ResNet-50 (default)")
        model = build_resnet50(num_classes)

    # Load weights
    try:
        model.load_state_dict(cleaned_state_dict)
        print("Weights loaded successfully")
    except RuntimeError as e:
        print(f"  Direct loading failed: {str(e)[:100]}")
        print(f"  Trying flexible loading...")

        model_dict = model.state_dict()
        matched = {
            k: v
            for k, v in cleaned_state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        print(f"  Matched: {len(matched)}/{len(model_dict)} layers")

        if len(matched) < len(model_dict) * 0.5:
            raise RuntimeError(
                f"Architecture mismatch: only {len(matched)}/{len(model_dict)} layers matched."
            )

        model_dict.update(matched)
        model.load_state_dict(model_dict)

    model = model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────
# FEATURE EXTRACTOR
# ──────────────────────────────────────────


class PlantNetFeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        model_name = type(base_model).__name__
        self._model_name = model_name
        self._use_timm = False

        if "EfficientNet" in model_name:
            self.base = base_model
            self.feature_dim = base_model.num_features
            self._use_timm = True

        elif "DenseNet" in model_name:
            self.base = base_model
            self.feature_dim = base_model.num_features
            self._use_timm = True

        elif "ResNet" in model_name:
            # Works for ResNet and Wide ResNet from both timm and torchvision
            if hasattr(base_model, "forward_features"):
                # timm model
                self.base = base_model
                self.feature_dim = base_model.num_features
                self._use_timm = True
            else:
                # torchvision model
                self.features = nn.Sequential(*list(base_model.children())[:-1])
                self.feature_dim = base_model.fc.in_features

        else:
            # Generic fallback — try timm style first
            if hasattr(base_model, "forward_features"):
                self.base = base_model
                self.feature_dim = getattr(base_model, "num_features", 2048)
                self._use_timm = True
            else:
                self.features = nn.Sequential(*list(base_model.children())[:-1])
                self.feature_dim = 2048

    def forward(self, x):
        if self._use_timm:
            x = self.base.forward_features(x)
            x = self.base.global_pool(x)
            return x.flatten(1)
        else:
            x = self.features(x)
            return x.flatten(1)


def get_feature_extractor(weights_tar_path, class_idx_to_species_path, device="cpu"):
    model = load_plantnet_model(weights_tar_path, class_idx_to_species_path, device)
    extractor = PlantNetFeatureExtractor(model)
    extractor = extractor.to(device)
    extractor.eval()
    return model, extractor


# ──────────────────────────────────────────
# INSPECTION UTILITY
# ──────────────────────────────────────────


def inspect_tar_checkpoint(tar_path):
    print(f"\nInspecting: {tar_path}")
    print("-" * 50)

    checkpoint = torch.load(tar_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        print(f"Type: dict with keys: {list(checkpoint.keys())}")

        for key in checkpoint.keys():
            val = checkpoint[key]
            if isinstance(val, dict):
                print(f"  '{key}': dict with {len(val)} entries")
                sample_keys = list(val.keys())[:3]
                for sk in sample_keys:
                    if hasattr(val[sk], "shape"):
                        print(f"    '{sk}': tensor {val[sk].shape}")
                    else:
                        print(f"    '{sk}': {type(val[sk]).__name__} = {val[sk]}")
            elif isinstance(val, (int, float, str)):
                print(f"  '{key}': {val}")
            elif hasattr(val, "shape"):
                print(f"  '{key}': tensor {val.shape}")
            else:
                print(f"  '{key}': {type(val).__name__}")

    print("-" * 50)
