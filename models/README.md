# Pretrained Model Weights

This directory holds PlantNet-300K pretrained model weights and metadata,
which are **not stored in this repository**. They are publicly distributed
by the Pl@ntNet team. Download them from the two sources below before
running the pipeline.

## Download Sources

### Model weights (4 files)

**https://seafile.plantnet.org/d/01ab6658dad6447c95ae/?p=%2F&mode=list**

Download:
- `resnet50_weights_best_acc.tar`
- `wide_resnet50_2_weights_best_acc.tar`
- `efficientnet_b0_weights_best_acc.tar`
- `densenet121_weights_best_acc.tar`

### Metadata files (3 files)

**https://lab.plantnet.org/seafile/d/bed81bc15e8944969cf6/**

Download:
- `class_idx_to_species_id.json`
- `plantnet300K_species_id_2_name.json`
- `plantnet300K_metadata.json`

Place all seven files directly in this `models/` directory.

## Verification

After placing the files in this directory:

```bash
python scripts/test_model_load.py
```

This script loads each architecture and confirms the weights are intact.

## Citation

If you use these weights, please cite the PlantNet-300K paper:

Garcin, C., Joly, A., Bonnet, P., Affouard, A., Lombardo, J.-C.,
Chouet, M., Servajean, M., Lorieul, T., & Salmon, J. (2021).
*Pl@ntNet-300K: A Plant Image Dataset with High Label Ambiguity and a
Long-Tailed Distribution.* NeurIPS Datasets and Benchmarks Track.