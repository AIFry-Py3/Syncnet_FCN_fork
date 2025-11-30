# SyncNet FCN - Training & Running Commands

## 🚀 Quick Start

### Prerequisites
```powershell
# Activate virtual environment
.\\venv\\Scripts\\Activate.ps1

# Verify CUDA is available
.\\venv\\Scripts\\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📊 Training the Model

### Start Fresh Training (Recommended)
```powershell
.\\venv\\Scripts\\python.exe train_syncnet_fcn_improved.py `
    --data_dir "E:/voxc2/vox2_dev_mp4_partaa~/dev/mp4" `
    --pretrained_model data/syncnet_v2.model `
    --batch_size 4 `
    --epochs 30 `
    --lr 0.00001 `
    --output_dir checkpoints_regression `
    --num_workers 2
```

### Resume from Old Checkpoint (NOT Recommended - Architecture Changed)
```powershell
# Old checkpoint won't work due to architecture change (classification → regression)
# If you try to resume from syncnet_fcn_epoch2.pth, you'll get shape mismatch errors
# Better to start fresh training
```

### Training with Attention
```powershell
.\\venv\\Scripts\\python.exe train_syncnet_fcn_improved.py `
    --data_dir "E:/voxc2/vox2_dev_mp4_partaa~/dev/mp4" `
    --pretrained_model data/syncnet_v2.model `
    --batch_size 2 `
    --epochs 30 `
    --lr 0.00001 `
    --output_dir checkpoints_attention `
    --use_attention `
    --num_workers 2
```

---

## 🎯 Testing the Model

### Test on Video File
```powershell
.\\venv\\Scripts\\python.exe test_sync_detection.py `
    --video test_videos/original.avi `
    --model data/syncnet_v2.model
```

### Test on HLS Stream
```powershell
.\\venv\\Scripts\\python.exe test_sync_detection.py `
    --hls http://example.com/stream.m3u8 `
    --model checkpoints_regression/syncnet_fcn_best.pth `
    --duration 10
```

---

## 📈 Model Architecture

### Key Changes (Regression Version)
- **Output**: `[B, 1, T]` continuous offset values (not probability distribution)
- **Range**: ±125 frames (±5 seconds at 25fps) for real-time streaming
- **Loss**: L1Loss (regression) instead of CrossEntropyLoss (classification)
- **Metrics**: MAE, RMSE, ±1 frame accuracy, ±1 second accuracy

### Expected Performance

| Metric | Initial (Epoch 1-2) | Mid Training (Epoch 10-15) | Target (Epoch 25-30) |
|--------|---------------------|----------------------------|----------------------|
| **MAE** | 30-40 frames | 10-15 frames | < 5 frames |
| **RMSE** | 40-50 frames | 15-20 frames | < 8 frames |
| **±1 Frame Acc** | 5-10% | 30-40% | > 50% |
| **±1 Second Acc** | 20-30% | 60-70% | > 85% |

---

## 🔧 Hyperparameter Tuning

### Reduce Batch Size (GPU OOM)
```powershell
--batch_size 2  # or even 1
--num_workers 0  # Disable multiprocessing
```

### Increase Learning Rate (Faster Convergence)
```powershell
--lr 0.0001  # 10x higher (use with caution)
```

### Decrease Learning Rate (More Stable)
```powershell
--lr 0.000001  # 10x lower
```

---

## 📁 Output Files

### Checkpoints Directory Structure
```
checkpoints_regression/
├── syncnet_fcn_improved_epoch1.pth
├── syncnet_fcn_improved_epoch2.pth
├── ...
└── syncnet_fcn_best.pth  # Best model based on ±1 second accuracy
```

### Checkpoint Contents
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `epoch`: Current epoch number
- `loss`: Training loss
- `metrics`: MAE, RMSE, accuracies

---

## 🐛 Troubleshooting

### Import Errors
```powershell
# Make sure you're in the right directory
cd C:\\Users\\admin\\Syncnet_FCN

# Activate venv
.\\venv\\Scripts\\Activate.ps1
```

### CUDA Out of Memory
```powershell
# Reduce batch size
--batch_size 1

# Or use CPU
$env:CUDA_VISIBLE_DEVICES = ""
```

### FFmpeg Not Found
```powershell
# Install FFmpeg
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### No Videos Found
```powershell
# Check your data directory path
# Should contain .mp4 files in subdirectories
ls E:/voxceleb2_dataset/VoxCeleb2/dev/**/*.mp4
```

---

## 📊 Monitoring Training

### Watch for These Signs

✅ **Good Training**:
- Loss decreasing steadily
- MAE decreasing over epochs
- ±1 second accuracy improving
- No "WARN" messages about failed samples

⚠️ **Warning Signs**:
- Loss increasing → Lower learning rate
- MAE stuck → Check data pipeline
- Many failed samples → Verify video files

### Example Good Training Output
```
Epoch 1 Summary:
  Total Loss: 15.2341
  MAE: 35.67 frames (1.427 seconds)
  ±1 Second Accuracy: 25.34%

Epoch 10 Summary:
  Total Loss: 5.8923
  MAE: 12.45 frames (0.498 seconds)
  ±1 Second Accuracy: 68.91%

Epoch 25 Summary:
  Total Loss: 2.1456
  MAE: 4.23 frames (0.169 seconds)
  ±1 Second Accuracy: 87.65%
  ✓ New best model saved!
```

---

## 🎬 Using Trained Model

### Load Best Model
```python
from SyncNetModel_FCN import StreamSyncFCN

# Load model
model = StreamSyncFCN(max_offset=125)
checkpoint = torch.load('checkpoints_regression/syncnet_fcn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process video
offset, confidence = model.process_video_file('test.mp4')
print(f"Offset: {offset:.2f} frames ({offset/25:.3f} seconds)")
```

### Real-Time Streaming
```python
# Process HLS stream
offset, conf = model.process_hls_stream(
    'http://example.com/live.m3u8',
    segment_duration=10
)

if abs(offset) > 25:  # More than 1 second out of sync
    print(f"⚠️ WARNING: {offset/25:.2f} seconds out of sync!")
```

---

## 📝 Summary

### What Changed from Original
| Aspect | Original | New (Regression) |
|--------|----------|------------------|
| **Task** | Binary classification | Continuous regression |
| **Output** | `[B, 31, T]` probabilities | `[B, 1, T]` offsets |
| **Range** | ±15 frames (0.6s) | ±125 frames (5s) |
| **Loss** | CrossEntropyLoss | L1Loss |
| **Metrics** | Accuracy | MAE, RMSE, tolerances |

### Why Regression is Better
1. **Continuous values**: Offset is inherently continuous
2. **Simpler model**: 1 output channel vs 251 classes
3. **Better for streaming**: ±5 second range handles network delays
4. **More interpretable**: Direct offset prediction in frames

---

## 🚀 Next Steps After Training

1. **Evaluate on test set**: Run on held-out videos
2. **Compare with baseline**: Test against original SyncNet
3. **Fine-tune if needed**: Unfreeze conv layers, train with lower LR
4. **Deploy**: Use for real-time stream monitoring

**Good luck with training!** 🎉
