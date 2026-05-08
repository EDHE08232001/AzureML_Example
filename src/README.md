# src/ — Source Package

This directory contains the MCUCoder source code as a Python package.

For full documentation, dataset download instructions, training and evaluation
guides, please see the **repository root README**:

```
ELG5378_Project/README.md
```

## Module Overview

| File | Responsibility |
|------|----------------|
| `config.py` | Absolute paths and all hyperparameters |
| `model.py` | `Encoder`, `Decoder`, `MCUCoder` (compressai-based) |
| `data.py` | `RecursiveImageDataset` and `build_dataloaders` |
| `losses.py` | `ProgressiveLoss` (MS-SSIM + MSE), `compute_psnr`, `compute_msssim_db` |
| `train.py` | Training loop, LR schedule, multi-rate validation, checkpointing |
| `evaluate.py` | RD evaluation with quantization, JPEG baseline, RD plot generation |
| `prepare_data.py` | ImageNet high-res selection, resize, noise, flat-copy |
| `utils.py` | `set_seed`, `get_device`, `format_metrics` |

## Running

Do **not** run files in this package directly.  Use the entry point at the
repository root:

```bash
# from ELG5378_Project/
python main.py
```
