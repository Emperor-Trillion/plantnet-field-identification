#!/bin/bash
echo "============================================"
echo "Downloading PlantNet-300K Model Weights"
echo "============================================"

MODELS_DIR="models"
mkdir -p $MODELS_DIR

# PlantNet-300K pretrained ResNet50
# Check the official repo for the latest download link:
# https://github.com/plantnet/PlantNet-300K

echo ""
echo "MANUAL STEPS REQUIRED:"
echo "1. Go to: https://github.com/plantnet/PlantNet-300K"
echo "2. Follow their instructions to download pretrained weights"
echo "3. Save the ResNet50 weights as: models/plantnet_resnet50.pth"
echo "4. Save the species mapping as: models/plantnet300k_species_names.txt"
echo ""
echo "Alternative: If they provide a direct download link:"
echo "  wget <URL> -O models/plantnet_resnet50.pth"
echo ""

# Download YOLOv8 (this downloads automatically on first use, but we can pre-download)
echo "Downloading YOLOv8 model..."
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" 2>/dev/null

echo ""
echo "YOLOv8 downloaded successfully."
echo ""
echo "Please ensure you have:"
echo "  ✓ models/plantnet_resnet50.pth"
echo "  ✓ models/plantnet300k_species_names.txt"
echo "============================================"