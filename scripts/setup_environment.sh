#!/bin/bash
echo "============================================"
echo "Setting Up Project Environment"
echo "============================================"

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Create all directories
mkdir -p data/raw_images
mkdir -p data/cropped_rois
mkdir -p data/features
mkdir -p data/csvs
mkdir -p models
mkdir -p outputs/figures
mkdir -p outputs/tables
mkdir -p paper/figures

# Create .gitkeep files for empty directories
touch data/raw_images/.gitkeep
touch data/cropped_rois/.gitkeep
touch data/features/.gitkeep
touch data/csvs/.gitkeep
touch outputs/figures/.gitkeep
touch outputs/tables/.gitkeep

echo ""
echo "Environment setup complete."
echo ""
echo "Next steps:"
echo "1. Place your images in data/raw_images/"
echo "2. Download model weights: bash scripts/download_models.sh"
echo "3. Update config.yaml with your API key"
echo "4. Run pipeline: python scripts/run_full_pipeline.py"
echo "============================================"