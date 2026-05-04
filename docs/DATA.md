# Data Access Policy

This repository contains pipeline code, configuration, output CSVs, and
generated figures — but no raw image data and no pretrained model weights.
This document describes how to obtain those.

## Pretrained Models — Public Download

The four PlantNet-300K pretrained architectures (ResNet-50, Wide ResNet-50-2,
EfficientNet-B0, DenseNet-121) are publicly distributed by the Pl@ntNet team.

**Source:** https://seafile.plantnet.org/d/01ab6658dad6447c95ae/?p=%2F&mode=list

Download the four `.tar` weight files and three metadata `.json` files and
place them in the `models/` directory. See [../models/README.md](../models/README.md)
for the full file list and verification steps.

## Field Image Dataset — Available on Request

The 303 raw field photographs used in this paper, along with downstream
artifacts (655 ROI crops, extracted feature matrices, similarity matrices)
are not publicly distributed. They are available on request for academic
research and reproduction purposes.

### How to Request

Email **sundayysanni@gmail.com** with:

- Your name and affiliation
- A short description of intended use (e.g., reproduction, follow-up research, teaching)
- Confirmation that the data will not be redistributed without permission

A response is typically sent within a few business days, with a download link
and a recommended directory layout matching this repository's expectations.

### What's Included in the Repository Without the Dataset

Even without the raw images, the repository contains:

- All pipeline source code (`src/`, `scripts/`)
- All result CSVs at every pipeline stage (`data/csvs/`, `outputs/tables/`)
- Cached external API responses (`data/csvs/_plantnet_api_cache.json`,
  `data/csvs/_gbif_taxonomy_cache.json`)
- Generated figures and analysis tables (`outputs/`)

This means all quantitative claims, tables, and figures in the paper can be
inspected, audited, and re-analyzed from the CSVs alone, without re-running
the image-processing stages.

### Why Restricted Access

The dataset includes opportunistically-collected field photographs whose
redistribution rights have not been universally cleared. Restricted access
ensures appropriate use without compromising the reproducibility goal of
the paper.

## Output CSVs — Always Public

The following are committed to this repository directly and require no request:

| File | Description |
|---|---|
| `data/csvs/03_flower_rois.csv` | ROI bounding boxes and crop paths |
| `data/csvs/05_feature_duplicates.csv` | Deduplicated feature pairs |
| `data/csvs/06_classification_results.csv` | Top-5 predictions per ROI |
| `data/csvs/_gbif_taxonomy_cache.json` | GBIF taxonomic lookups |
| `data/csvs/_plantnet_api_cache.json` | PlantNet API responses |
| `outputs/tables/combined_knn_crossval.csv` | KNN-CV summary across all models |
| `outputs/tables/<model>_*.csv` | Per-model intermediate and final results |