# FCN-SyncNet: Fully Convolutional Audio-Video Synchronization Network

> ⚠️ **Research Prototype**: This is an experimental extension of SyncNet exploring fully convolutional architectures for extended-range audio-video sync detection. See [Current Status](#current-status) for details.

This repository contains an enhanced **Fully Convolutional SyncNet (FCN-SyncNet)** for audio-video synchronization detection. Built upon the original SyncNet architecture, this version introduces:

- 🔧 **Fully Convolutional Architecture** - No FC layers, supports variable-length inputs
- 📊 **Temporal Feature Maps** - Dense frame-by-frame sync predictions
- 🔗 **Correlation-based Fusion** - Audio-video temporal correlation layer
- ⚡ **Transfer Learning** - Leverages pretrained SyncNet weights
- 🎯 **Extended Offset Range** - Targets ±5 seconds (±125 frames) detection range
- 🆕 **Classification Approach** - 251-class classification to avoid regression-to-mean

## Current Status

### ✅ What Works
- **Original SyncNet**: Accurate offset detection within ±15 frames (recommended for production)
- **FCN Architecture**: Successfully loads and runs inference
- **Transfer Learning**: Pretrained SyncNet weights properly loaded
- **CLI Tool**: `detect_sync.py` for quick video analysis
- **Demo Script**: `generate_demo.py` for comparison demonstrations

### ⚠️ Known Limitations
1. **Regression-to-Mean**: Initial regression approach with MSE/L1 loss caused model to predict mean offset
2. **Training Time**: VoxCeleb2 training requires ~11 hours per epoch (132K videos)
3. **Classification Training**: Not yet completed due to time constraints

### 🔬 Research Findings
1. **Transfer learning is effective**: FCN with pretrained weights learns meaningful audio-video features
2. **Regression fails at scale**: More training epochs = worse performance due to regression-to-mean
3. **Classification recommended**: Treating offset detection as 251-class classification avoids mean regression
4. **Early stopping important**: Epoch 2 outperformed epoch 21 in regression experiments

### 📋 Two Approaches Implemented

#### 1. Regression Approach (Original)
- File: `SyncNetModel_FCN.py`, `train_syncnet_fcn_improved.py`
- Issue: Regression-to-mean problem
- Mitigation: Linear calibration layer

#### 2. Classification Approach (Recommended)
- File: `SyncNetModel_FCN_Classification.py`, `train_syncnet_fcn_classification.py`
- 251 classes for offsets -125 to +125 frames (±5 seconds)
- Uses CrossEntropyLoss with label smoothing
- Training not yet completed

## Features Comparison

| Feature | Original SyncNet | FCN-SyncNet (Regression) | FCN-SyncNet (Classification) |
|---------|------------------|--------------------------|------------------------------|
| Architecture | FC layers | Fully Convolutional | Fully Convolutional |
| Input Length | Fixed window | Variable length | Variable length |
| Output | Single offset | Continuous regression | 251 discrete classes |
| Max Offset | ±15 frames ✅ | ±125 frames | ±125 frames (±5 sec) |
| Loss Function | N/A | L1/MSE | CrossEntropy |
| Transfer Learning | ❌ | ✅ | ✅ |
| Production Ready | ✅ | ⚠️ Research | ⚠️ Research |

## Quick Start

### For Production: Use Original SyncNet
For reliable offset detection within ±15 frames:
```bash
python demo_syncnet.py --videofile path/to/video.avi --tmp_dir data/work/pytmp
```

### CLI Tool (FCN with Calibration)
```bash
python detect_sync.py video.mp4              # Basic usage
python detect_sync.py video.mp4 --verbose    # Detailed output
```

### Demo Script (Comparison)
```bash
python generate_demo.py --compare --cleanup  # Compare FCN vs Original SyncNet
```

## Installation

### Dependencies
```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch >= 1.4.0
- OpenCV
- FFmpeg (must be in PATH)
- python_speech_features
- scipy, numpy

### Download Pretrained Model
```bash
sh download_model.sh
```
This downloads `data/syncnet_v2.model` - the pretrained base model for transfer learning.

## Quick Start

### 1. Test on a Single Video (Original SyncNet - Recommended)
```bash
python demo_syncnet.py --videofile data/example.avi --tmp_dir data/work/pytmp
```

### 2. Test FCN Model
```bash
python test_sync_detection.py --video path/to/video.mp4 --model checkpoints/syncnet_fcn_epoch2.pth
```

### 3. FCN Pipeline (Experimental)
```bash
python run_fcn_pipeline.py --video path/to/video.mp4 --pretrained data/syncnet_v2.model
```

## Training

### Train Classification Model (Recommended)
```bash
python train_syncnet_fcn_classification.py \
    --data_dir /path/to/VoxCeleb2/dev/mp4 \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-4 \
    --max_offset 125
```

### Train Regression Model (Has regression-to-mean issue)
```bash
python train_syncnet_fcn_improved.py \
    --data_dir /path/to/VoxCeleb2/dev \
    --pretrained_model data/syncnet_v2.model \
    --batch_size 2 \
    --epochs 30 \
    --lr 0.00001 \
    --output_dir checkpoints_regression \
    --max_offset 125 \
    --unfreeze_epoch 10
```

**Training Notes:**
- VoxCeleb2 has ~132K videos, training is slow (~11 hours/epoch)
- Classification approach avoids regression-to-mean problem
- Early stopping recommended for regression approach

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Path to VoxCeleb2 dataset |
| `--pretrained_model` | `data/syncnet_v2.model` | Pretrained SyncNet weights |
| `--batch_size` | 4 | Batch size |
| `--epochs` | 20 | Number of training epochs |
| `--lr` | 0.00001 | Learning rate |
| `--max_offset` | 125 | Max offset in frames (125 = ±5 seconds) |
| `--unfreeze_epoch` | 10 | Epoch to unfreeze pretrained layers |
| `--use_attention` | False | Use cross-modal attention model |

## Evaluation

### Evaluate Trained Model
```bash
# Quick test on single video
python evaluate_model.py --model checkpoints_regression/syncnet_fcn_best.pth --video data/example.avi

# Evaluate on dataset
python evaluate_model.py \
    --model checkpoints_regression/syncnet_fcn_best.pth \
    --data_dir /path/to/VoxCeleb2/dev \
    --num_samples 500 \
    --full_report \
    --readme
```

## Model Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Video Frames   │     │   Audio MFCC    │
│ [B, 3, T, H, W] │     │ [B, 1, 13, T]   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  3D Conv Layers │     │  2D Conv Layers │
│  + Attention    │     │  + Attention    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Video Features  │     │ Audio Features  │
│   [B, 512, T']  │     │   [B, 512, T']  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
           ┌─────────────────┐
           │   Temporal      │
           │  Correlation    │
           └────────┬────────┘
                    ▼
           ┌─────────────────┐
           │ Offset Regressor│
           └────────┬────────┘
                    ▼
           ┌─────────────────┐
           │ Predicted Offset│
           │   [B, 1, T'']   │
           └─────────────────┘
```

## Project Structure

```
Syncnet_FCN/
├── SyncNetModel_FCN.py              # FCN regression model
├── SyncNetModel_FCN_Classification.py # FCN classification model (recommended)
├── SyncNetModel.py                  # Original SyncNet model
├── train_syncnet_fcn_classification.py # Classification training script
├── train_syncnet_fcn_improved.py    # Regression training script
├── detect_sync.py                   # CLI tool for sync detection
├── generate_demo.py                 # Demo comparison script
├── demo_syncnet.py                  # Original SyncNet demo
├── requirements.txt                 # Dependencies
├── data/
│   ├── syncnet_v2.model             # Pretrained base model
│   └── example.avi                  # Example video
├── checkpoints/                     # FCN checkpoints (regression, early)
├── checkpoints_regression/          # FCN checkpoints (regression, full)
├── checkpoints_classification/      # FCN checkpoints (classification)
└── detectors/                       # Face detection (S3FD)
```

## Key Files

| File | Description |
|------|-------------|
| `SyncNetModel_FCN_Classification.py` | **Recommended**: Classification-based FCN (251 classes) |
| `SyncNetModel_FCN.py` | Regression-based FCN with calibration |
| `train_syncnet_fcn_classification.py` | Training script for classification model |
| `detect_sync.py` | CLI tool for quick video analysis |
| `generate_demo.py` | Demo script comparing FCN vs Original SyncNet |

## Publications

Original SyncNet paper:
```bibtex
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```

## Contributing

This is a research prototype. Contributions exploring alternative approaches are welcome:
- Classification-based offset detection
- Contrastive learning methods
- Alternative loss functions for regression
- Multi-scale temporal modeling

## License

See [LICENSE.md](LICENSE.md) for details.

## Acknowledgments

- Original SyncNet implementation by VGG, University of Oxford
- VoxCeleb2 dataset for training data
