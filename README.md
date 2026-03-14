# BraTS 2020 Brain Tumor Segmentation

2.5D UNet trained on BraTS 2020 for whole-tumor segmentation with MC-Dropout uncertainty estimation.

**Test Dice: 0.8651** 

## Overview

Trains a 2.5D UNet on BraTS 2020 to segment whole-tumor regions from MRI scans. Uses MC-Dropout at inference to generate uncertainty maps showing where the model is least confident

## Architecture

- Input: 6 channels (FLAIR + T1ce × 3 context slices each)
- UNet Encoder widths: [16, 32, 64, 128, 256]
- Loss: DiceCELoss (MONAI)
- Optimizer: AdamW, lr=1e-4, cosine annealing


## Uncertainty Estimation

MC-Dropout with 20 forward passes. High-entropy regions (usually tumor boundaries) show where the model is least confident

## Results

| Split | Patients | Dice |
|-------|----------|------|
| Val   | 55       | 0.8606 |
| Test  | 56       | 0.8651 |

Per-patient Dice varies (~0.30–0.95) small tumors score lower

## Dataset

BraTS 2020 via [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) — 369 patients, FLAIR and T1ce modalities, expert-annotated whole-tumor masks.

Split: 70% train / 15% val / 15% test, patient-level, seed=42.

## Project Structure
```
src/
  data/         — dataset, preprocessing, splits, transforms
  models/       — UNet definition
  training/     — trainer, loss, metrics, train/evaluate scripts
  inference/    — predict, uncertainty
app/
  streamlit_app.py   — interactive demo
configs/
  config.yaml        — all hyperparameters
```

## Setup
```bash
git clone https://github.com/yourusername/brats-segmentation
cd brats-segmentation
pip install -r requirements.txt
cp .env.example .env  # set DATA_ROOT, PREPROCESSED_DIR, CHECKPOINT_DIR, SPLITS_DIR
```

## Usage

**Preprocess** (run once):
```bash
python -m src.data.preprocess
```

**Train:**
```bash
python -m src.training.train           # full training
python -m src.training.train --debug   # 10 patients, 20 epochs
```

**Evaluate:**
```bash
python -m src.training.evaluate
```

**Demo:**
```bash
streamlit run app/streamlit_app.py
```

## Limitations

- Binary segmentation only (whole tumor), not sub-regions
- 2.5D so no long-range 3D context
- Not benchmarked against nnU-Net

## Environment

Python 3.11 · PyTorch · MONAI · Streamlit 
```
