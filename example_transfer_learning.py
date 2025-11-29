#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Example: Transfer Learning with SyncNetFCN

This script demonstrates how to:
1. Load pretrained SyncNet weights into FCN model
2. Freeze conv layers and train new layers
3. Fine-tune entire model
4. Use different learning rates for different layer groups

Author: Demo
Date: 2025-11-29
"""

import torch
import torch.nn as nn
import torch.optim as optim
from SyncNetModel_FCN import SyncNetFCN, SyncNetFCN_WithAttention
from transfer_learning_utils import (
    load_pretrained_syncnet,
    freeze_pretrained_layers,
    unfreeze_all_layers,
    print_trainable_summary,
    get_optimizer_param_groups
)


def example_1_basic_transfer_learning():
    """
    Example 1: Load pretrained weights, freeze conv layers, train new layers.
    This is the recommended starting point.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Transfer Learning (Frozen Conv Layers)")
    print("="*80)
    
    # Create FCN model
    model = SyncNetFCN(embedding_dim=512, max_offset=15)
    
    # Load pretrained weights from original SyncNet
    # Replace with your actual checkpoint path
    pretrained_path = 'data/syncnet_v2.model'
    
    try:
        stats = load_pretrained_syncnet(
            fcn_model=model,
            pretrained_syncnet_path=pretrained_path,
            freeze_conv_layers=True,  # Freeze loaded conv layers
            verbose=True
        )
        
        print("\n✓ Transfer learning successful!")
        print(f"  Audio conv layers loaded: {stats['audio_conv_loaded']}")
        print(f"  Video conv layers loaded: {stats['video_conv_loaded']}")
        
    except FileNotFoundError:
        print(f"\n⚠ Checkpoint not found at: {pretrained_path}")
        print("  Please provide the correct path to your pretrained SyncNet model.")
        print("  For now, we'll continue with random initialization for demonstration.")
        
        # Manually freeze conv_layers for demo
        freeze_pretrained_layers(model, verbose=True)
    
    # Create optimizer - only new layers will be trained
    # Conv layers are frozen, so they won't receive gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=1e-3)
    
    print(f"\nOptimizer created with {len(trainable_params)} parameter groups")
    print("Ready to train! Conv layers frozen, new layers trainable.")
    
    return model, optimizer


def example_2_fine_tuning():
    """
    Example 2: Load pretrained weights, then fine-tune ALL layers.
    Use this after initial training with frozen layers.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Fine-Tuning All Layers")
    print("="*80)
    
    # Create and load model (same as example 1)
    model = SyncNetFCN(embedding_dim=512, max_offset=15)
    pretrained_path = 'data/syncnet_v2.model'
    
    try:
        load_pretrained_syncnet(
            fcn_model=model,
            pretrained_syncnet_path=pretrained_path,
            freeze_conv_layers=True,  # Initially freeze
            verbose=False  # Less verbose
        )
    except FileNotFoundError:
        print(f"⚠ Using random initialization (checkpoint not found)")
    
    # Unfreeze all layers for fine-tuning
    print("\nUnfreezing all layers for fine-tuning...")
    unfreeze_all_layers(model, verbose=True)
    
    # Use smaller learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("\nReady for fine-tuning with all layers trainable!")
    
    return model, optimizer


def example_3_differential_learning_rates():
    """
    Example 3: Use different learning rates for pretrained vs new layers.
    Pretrained conv layers get smaller LR, new layers get larger LR.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Differential Learning Rates")
    print("="*80)
    
    # Create and load model
    model = SyncNetFCN(embedding_dim=512, max_offset=15)
    pretrained_path = 'data/syncnet_v2.model'
    
    try:
        load_pretrained_syncnet(
            fcn_model=model,
            pretrained_syncnet_path=pretrained_path,
            freeze_conv_layers=False,  # Don't freeze - we'll use different LRs
            verbose=False
        )
    except FileNotFoundError:
        print(f"⚠ Using random initialization (checkpoint not found)")
    
    # Create parameter groups with different learning rates
    print("\nSetting up differential learning rates...")
    param_groups = get_optimizer_param_groups(
        fcn_model=model,
        pretrained_lr=1e-5,   # Small LR for pretrained conv layers
        new_layers_lr=1e-3    # Large LR for new layers
    )
    
    optimizer = optim.Adam(param_groups)
    
    print("\nOptimizer configured with 2 parameter groups.")
    print("Pretrained layers will update slowly, new layers will update quickly.")
    
    return model, optimizer


def example_4_progressive_unfreezing():
    """
    Example 4: Progressive unfreezing strategy.
    1. Train new layers with frozen conv
    2. Unfreeze and fine-tune all layers
    
    This would be used across multiple training phases.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Progressive Unfreezing Strategy")
    print("="*80)
    
    # Phase 1: Train with frozen conv layers
    print("\n--- PHASE 1: Train new layers (conv frozen) ---")
    model = SyncNetFCN(embedding_dim=512, max_offset=15)
    
    try:
        load_pretrained_syncnet(
            fcn_model=model,
            pretrained_syncnet_path='data/syncnet_v2.model',
            freeze_conv_layers=True,
            verbose=False
        )
    except FileNotFoundError:
        freeze_pretrained_layers(model, verbose=False)
    
    print_trainable_summary(model)
    
    # Train here with this optimizer...
    optimizer_phase1 = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    
    print("\nIn Phase 1, you would train for several epochs...")
    print("  → New layers learn to work with pretrained features")
    
    # Phase 2: Unfreeze and fine-tune
    print("\n--- PHASE 2: Fine-tune all layers ---")
    unfreeze_all_layers(model, verbose=False)
    print_trainable_summary(model)
    
    # Use smaller LR for fine-tuning
    optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-4)
    
    print("\nIn Phase 2, you would continue training...")
    print("  → All layers adapt together for optimal performance")
    
    return model, optimizer_phase2


def example_5_attention_model():
    """
    Example 5: Transfer learning with SyncNetFCN_WithAttention.
    Same process works for the attention-based variant.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Transfer Learning with Attention Model")
    print("="*80)
    
    # Create attention-based FCN model
    model = SyncNetFCN_WithAttention(embedding_dim=512, max_offset=15)
    
    try:
        stats = load_pretrained_syncnet(
            fcn_model=model,
            pretrained_syncnet_path='data/syncnet_v2.model',
            freeze_conv_layers=True,
            verbose=False
        )
        print(f"✓ Loaded {stats['audio_conv_loaded'] + stats['video_conv_loaded']} conv layers")
    except FileNotFoundError:
        print("⚠ Using random initialization")
        freeze_pretrained_layers(model, verbose=False)
    
    print_trainable_summary(model)
    
    # The attention layers are new, so they're trainable
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    
    print("\nAttention model ready!")
    print("  → Conv layers: pretrained (frozen)")
    print("  → Attention layers: random init (trainable)")
    print("  → Other new layers: random init (trainable)")
    
    return model, optimizer


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRANSFER LEARNING EXAMPLES FOR SYNCNETFCN")
    print("="*80)
    print("\nThese examples show different transfer learning strategies.")
    print("Choose the one that fits your training pipeline.\n")
    
    # Run all examples
    print("\n" + "#"*80)
    model1, opt1 = example_1_basic_transfer_learning()
    
    print("\n" + "#"*80)
    model2, opt2 = example_2_fine_tuning()
    
    print("\n" + "#"*80)
    model3, opt3 = example_3_differential_learning_rates()
    
    print("\n" + "#"*80)
    model4, opt4 = example_4_progressive_unfreezing()
    
    print("\n" + "#"*80)
    model5, opt5 = example_5_attention_model()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nRecommended approach:")
    print("  1. Start with Example 1 (frozen conv, train new layers)")
    print("  2. After convergence, move to Example 2 (fine-tune all layers)")
    print("  3. Or use Example 4 (progressive unfreezing) for best results")
    print("\nFor actual training, integrate chosen example into your training script.")
    print("="*80 + "\n")
