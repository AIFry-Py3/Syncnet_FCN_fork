# Transfer Learning Guide for SyncNetFCN

## Overview

This guide explains how to use **transfer learning** to load pretrained weights from your original SyncNet model into the new SyncNetFCN model.

## What You Get

✓ **Pretrained conv layers**: Audio and video encoder conv layers initialized from SyncNet  
✓ **Frozen backbones**: Conv layers frozen to preserve learned features  
✓ **Trainable new layers**: Correlation, predictor, and attention layers ready to train  
✓ **Flexible fine-tuning**: Easily unfreeze layers later for end-to-end training  

---

## Quick Start

### Step 1: Import the utility

```python
from transfer_learning_utils import load_pretrained_syncnet
from SyncNetModel_FCN import SyncNetFCN
```

### Step 2: Create your FCN model

```python
model = SyncNetFCN(embedding_dim=512, max_offset=15)
```

### Step 3: Load pretrained weights

```python
stats = load_pretrained_syncnet(
    fcn_model=model,
    pretrained_syncnet_path='data/syncnet_v2.model',
    freeze_conv_layers=True,
    verbose=True
)
```

### Step 4: Create optimizer (only trainable params)

```python
import torch.optim as optim

optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3
)
```

### Step 5: Train!

```python
# Your training loop here
for epoch in range(num_epochs):
    for audio_batch, video_batch, labels in dataloader:
        optimizer.zero_grad()
        sync_probs, _, _ = model(audio_batch, video_batch)
        loss = compute_loss(sync_probs, labels)
        loss.backward()
        optimizer.step()
```

---

## How It Works

### Layer Mapping

The `load_pretrained_syncnet` function automatically maps layers:

**Audio Branch:**
```
Original SyncNet          →  FCN Model
────────────────────────────────────────────────
netcnnaud.0.weight        →  audio_encoder.conv_layers.0.weight
netcnnaud.1.weight        →  audio_encoder.conv_layers.1.weight
netcnnaud.1.bias          →  audio_encoder.conv_layers.1.bias
...                       →  ...
netcnnaud.N.weight        →  audio_encoder.conv_layers.N.weight
```

**Video Branch:**
```
Original SyncNet          →  FCN Model
────────────────────────────────────────────────
netcnnlip.0.weight        →  video_encoder.conv_layers.0.weight
netcnnlip.1.weight        →  video_encoder.conv_layers.1.weight
netcnnlip.1.bias          →  video_encoder.conv_layers.1.bias
...                       →  ...
netcnnlip.N.weight        →  video_encoder.conv_layers.N.weight
```

**Skipped Layers:**
- `netfcaud.*` (FC layers) - Not compatible with FCN architecture
- `netfclip.*` (FC layers) - Not compatible with FCN architecture

### What Gets Frozen

When `freeze_conv_layers=True`:

✓ **Frozen** (no gradients):
- `audio_encoder.conv_layers.*`  
- `video_encoder.conv_layers.*`

✓ **Trainable** (receives gradients):
- `audio_encoder.channel_conv.*`  
- `audio_encoder.channel_attn.*`  
- `video_encoder.channel_conv.*`  
- `video_encoder.channel_attn.*`  
- `correlation.*`  
- `sync_predictor.*`  
- `temporal_smoother.*`

---

## Training Strategies

### Strategy 1: Frozen Conv (Recommended Start)

**When**: First training phase  
**Goal**: Train new layers while preserving pretrained features

```python
from transfer_learning_utils import load_pretrained_syncnet

model = SyncNetFCN(embedding_dim=512, max_offset=15)

# Load and freeze conv layers
load_pretrained_syncnet(
    fcn_model=model,
    pretrained_syncnet_path='data/syncnet_v2.model',
    freeze_conv_layers=True,
    verbose=True
)

# Optimizer for trainable params only
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3  # Larger LR for new layers
)

# Train for several epochs...
```

**Expected Output:**
```
================================================================================
TRANSFER LEARNING: Loading pretrained SyncNet weights into FCN model
================================================================================
✓ Loaded pretrained checkpoint from: data/syncnet_v2.model
  Total parameters in checkpoint: 156

--------------------------------------------------------------------------------
AUDIO ENCODER: Mapping netcnnaud.* → audio_encoder.conv_layers.*
--------------------------------------------------------------------------------
  ✓ netcnnaud.0.weight → audio_encoder.conv_layers.0.weight [64, 1, 3, 3]
  ✓ netcnnaud.1.weight → audio_encoder.conv_layers.1.weight [64]
  ...

--------------------------------------------------------------------------------
VIDEO ENCODER: Mapping netcnnlip.* → video_encoder.conv_layers.*
--------------------------------------------------------------------------------
  ✓ netcnnlip.0.weight → video_encoder.conv_layers.0.weight [96, 3, 5, 7, 7]
  ✓ netcnnlip.1.weight → video_encoder.conv_layers.1.weight [96]
  ...

================================================================================
TRANSFER LEARNING SUMMARY
================================================================================
Audio conv layers loaded:   21
Audio conv layers skipped:  0
Video conv layers loaded:   21
Video conv layers skipped:  0
FC layers skipped:          8
================================================================================

Freezing loaded conv layers...
  Frozen parameters:    12,345,678
  Trainable parameters: 2,345,678

================================================================================
MODEL PARAMETER SUMMARY
================================================================================
Frozen parameters:     12,345,678 (84.1%)
Trainable parameters:  2,345,678 (15.9%)
Total parameters:      14,691,356
================================================================================
```

---

### Strategy 2: Fine-Tuning (After Initial Training)

**When**: After strategy 1 converges  
**Goal**: Adapt all layers end-to-end

```python
from transfer_learning_utils import unfreeze_all_layers

# Unfreeze all layers
unfreeze_all_layers(model, verbose=True)

# Use smaller LR for fine-tuning
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4  # Smaller LR to avoid disrupting pretrained features
)

# Continue training...
```

---

### Strategy 3: Differential Learning Rates

**When**: You want to train all layers from the start  
**Goal**: Update pretrained layers slowly, new layers quickly

```python
from transfer_learning_utils import get_optimizer_param_groups

model = SyncNetFCN(embedding_dim=512, max_offset=15)

# Load without freezing
load_pretrained_syncnet(
    fcn_model=model,
    pretrained_syncnet_path='data/syncnet_v2.model',
    freeze_conv_layers=False,  # Don't freeze!
    verbose=True
)

# Different LRs for different layer groups
param_groups = get_optimizer_param_groups(
    fcn_model=model,
    pretrained_lr=1e-5,   # Very small for pretrained conv
    new_layers_lr=1e-3    # Larger for new layers
)

optimizer = optim.Adam(param_groups)

# Train...
```

---

### Strategy 4: Progressive Unfreezing

**When**: You want maximum stability  
**Goal**: Gradually unfreeze layers during training

```python
# Phase 1: Train new layers only (epochs 0-10)
model = SyncNetFCN(embedding_dim=512, max_offset=15)
load_pretrained_syncnet(model, 'data/syncnet_v2.model', freeze_conv_layers=True)
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

# Train for 10 epochs...

# Phase 2: Fine-tune all layers (epochs 11-20)
unfreeze_all_layers(model)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Continue training for 10 more epochs...
```

---

## Working with Attention Model

The same process works for `SyncNetFCN_WithAttention`:

```python
from SyncNetModel_FCN import SyncNetFCN_WithAttention
from transfer_learning_utils import load_pretrained_syncnet

# Create attention model
model = SyncNetFCN_WithAttention(embedding_dim=512, max_offset=15)

# Load pretrained conv weights (attention layers remain random)
load_pretrained_syncnet(
    fcn_model=model,
    pretrained_syncnet_path='data/syncnet_v2.model',
    freeze_conv_layers=True,
    verbose=True
)

# The attention layers are new and will be trainable
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3
)
```

---

## Troubleshooting

### Issue: "Shape mismatch" warnings

**Cause**: Your FCN model's conv layers have different shapes than original SyncNet  
**Solution**: This is expected for some layers (e.g., different kernel sizes). Only matching layers are loaded.

### Issue: "Checkpoint not found"

**Cause**: Wrong path to pretrained SyncNet model  
**Solution**: Verify the path:

```python
import os
checkpoint_path = 'data/syncnet_v2.model'
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found at: {checkpoint_path}")
    print(f"Please update the path!")
```

### Issue: Model not learning

**Possible causes**:
1. All layers are frozen → Use `unfreeze_all_layers` or check `requires_grad`
2. Learning rate too small → Increase for new layers (1e-3 is good)
3. Learning rate too large for fine-tuning → Use 1e-4 or 1e-5

**Debug**:
```python
from transfer_learning_utils import print_trainable_summary

print_trainable_summary(model)
# Check that expected layers are trainable
```

---

## Integration with Existing Code

### In SyncNetInstance_FCN

Modify the `__init__` method to support transfer learning:

```python
class SyncNetInstance_FCN(torch.nn.Module):
    def __init__(self, model_type='fcn', embedding_dim=512, max_offset=15, 
                 use_attention=False, pretrained_syncnet_path=None):
        super(SyncNetInstance_FCN, self).__init__()
        
        # ... existing code ...
        
        # Transfer learning
        if pretrained_syncnet_path is not None:
            from transfer_learning_utils import load_pretrained_syncnet
            print(f"Loading pretrained SyncNet from: {pretrained_syncnet_path}")
            load_pretrained_syncnet(
                fcn_model=self.model,
                pretrained_syncnet_path=pretrained_syncnet_path,
                freeze_conv_layers=True,
                verbose=True
            )
```

Usage:
```python
syncnet = SyncNetInstance_FCN(
    use_attention=False,
    pretrained_syncnet_path='data/syncnet_v2.model'
)
```

### In Training Script

```python
import torch
from SyncNetModel_FCN import SyncNetFCN
from transfer_learning_utils import load_pretrained_syncnet

def main():
    # Create model
    model = SyncNetFCN(embedding_dim=512, max_offset=15)
    
    # Transfer learning
    load_pretrained_syncnet(
        fcn_model=model,
        pretrained_syncnet_path='checkpoints/syncnet_v2.model',
        freeze_conv_layers=True,
        verbose=True
    )
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Your training code here
            pass

if __name__ == '__main__':
    main()
```

---

## Summary

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_pretrained_syncnet()` | Load conv weights from SyncNet → FCN |
| `freeze_pretrained_layers()` | Freeze conv layers |
| `unfreeze_all_layers()` | Unfreeze for fine-tuning |
| `get_optimizer_param_groups()` | Different LRs for different layers |
| `print_trainable_summary()` | See what's frozen/trainable |

**Recommended Workflow:**

1. **Phase 1** (10-20 epochs): Load pretrained weights, freeze conv, train new layers (lr=1e-3)
2. **Phase 2** (10-20 epochs): Unfreeze all, fine-tune end-to-end (lr=1e-4)
3. **Phase 3** (optional): Lower LR to 1e-5 for final refinement

**Files Created:**

- `transfer_learning_utils.py` - Core utilities
- `example_transfer_learning.py` - Usage examples

---

## Next Steps

Once you have transfer learning working:

1. **Add checkpointing** to save/resume training
2. **Implement StreamSync** for streaming/real-time inference
3. **Experiment with learning rates** to find optimal values for your dataset

Good luck with your project! 🚀
