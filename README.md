## Data Access

This repository contains code only. The dataset and pretrained models are not bundled.

### Pretrained Models (publicly available)

The four PlantNet-300K pretrained model weights used in this work are
distributed by the Pl@ntNet team and available at:

**https://seafile.plantnet.org/d/01ab6658dad6447c95ae/?p=%2F&mode=list**

Download all four `.tar` files plus the three metadata JSONs and place them
in `models/`. See [models/README.md](models/README.md) for the exact list.

### Field Image Dataset (available on request)

The 303 raw field photographs and downstream artifacts (ROI crops, feature
matrices) used in this paper are **not publicly distributed**. Access is
granted on request for research and reproducibility purposes.

To request the dataset, please email:

**sundayysanni@gmail.com**

with a brief description of intended use. The dataset will be sent as a
single archive via cloud storage link. Pipeline output CSVs are included
in the repository under `data/csvs/` and `outputs/tables/` so that all
quantitative results in the paper can be inspected without the raw images.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Emperor-Trillion/plantnet-field-identification.git
cd plantnet-field-identification

# 2. Environment
python3.11 -m venv .virtualEnv
source .virtualEnv/bin/activate
pip install -r requirements.txt

# 3. Models (public)
# Download from: https://seafile.plantnet.org/d/01ab6658dad6447c95ae/?p=%2F&mode=list
# Place all .tar weights and .json metadata files in models/

# 4. Dataset (on request)
# Email sundayysanni@gmail.com for access.
# Place received contents into data/raw_images/

# 5. Configure
# Edit config.yaml — primarily the PlantNet API key for Stage 4 validation

# 6. Run
python -m scripts.run_full_pipeline
python -m scripts.compare_models
python scripts/multi_model_knn_cv.py
```