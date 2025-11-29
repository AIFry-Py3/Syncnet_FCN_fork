#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Transfer Learning Utilities for SyncNetFCN

This module provides functions to load pretrained SyncNet weights
into the FCN-based SyncNet model, with proper layer mapping and freezing.

Key functions:
1. load_pretrained_syncnet: Load conv weights from original SyncNet
2. freeze_pretrained_layers: Freeze loaded conv layers
3. print_transfer_summary: Show what was loaded and what's trainable

Author: Enhanced version
Date: 2025-11-29
"""

import torch
import torch.nn as nn
from collections import OrderedDict


def load_pretrained_syncnet(fcn_model, pretrained_syncnet_path, freeze_conv_layers=True, verbose=True):
    """
    Load pretrained conv weights from original SyncNet into FCN model.
    
    This function:
    1. Loads the original SyncNet checkpoint
    2. Maps audio conv layers: netcnnaud.* → audio_encoder.conv_layers.*
    3. Maps video conv layers: netcnnlip.* → video_encoder.conv_layers.*
    4. Skips FC layers (netfcaud, netfclip)
    5. Optionally freezes loaded layers
    
    Args:
        fcn_model: SyncNetFCN or SyncNetFCN_WithAttention instance
        pretrained_syncnet_path: Path to original SyncNet checkpoint (.model file)
        freeze_conv_layers: If True, freeze loaded conv layers (default: True)
        verbose: Print detailed loading information
        
    Returns:
        transfer_stats: Dictionary with loading statistics
    """
    
    if verbose:
        print("="*80)
        print("TRANSFER LEARNING: Loading pretrained SyncNet weights into FCN model")
        print("="*80)
    
    # Load pretrained checkpoint
    try:
        pretrained_state = torch.load(pretrained_syncnet_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(pretrained_state, dict):
            if 'model_state_dict' in pretrained_state:
                pretrained_dict = pretrained_state['model_state_dict']
            elif 'state_dict' in pretrained_state:
                pretrained_dict = pretrained_state['state_dict']
            else:
                pretrained_dict = pretrained_state
        else:
            # It's a model object
            pretrained_dict = pretrained_state.state_dict()
            
        if verbose:
            print(f"✓ Loaded pretrained checkpoint from: {pretrained_syncnet_path}")
            print(f"  Total parameters in checkpoint: {len(pretrained_dict)}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load pretrained checkpoint: {e}")
    
    # Get FCN model's state dict
    fcn_state_dict = fcn_model.state_dict()
    
    # Statistics tracking
    stats = {
        'audio_conv_loaded': 0,
        'video_conv_loaded': 0,
        'audio_conv_skipped': 0,
        'video_conv_skipped': 0,
        'fc_layers_skipped': 0,
        'unmatched_pretrained': 0,
        'loaded_param_names': [],
        'skipped_param_names': []
    }
    
    # Create mapping for audio encoder
    if verbose:
        print("\n" + "-"*80)
        print("AUDIO ENCODER: Mapping netcnnaud.* → audio_encoder.conv_layers.*")
        print("-"*80)
    
    audio_mapping = create_audio_mapping(pretrained_dict, fcn_state_dict, verbose)
    
    # Create mapping for video encoder
    if verbose:
        print("\n" + "-"*80)
        print("VIDEO ENCODER: Mapping netcnnlip.* → video_encoder.conv_layers.*")
        print("-"*80)
    
    video_mapping = create_video_mapping(pretrained_dict, fcn_state_dict, verbose)
    
    # Apply mappings
    new_state_dict = fcn_state_dict.copy()
    
    for pretrained_key, fcn_key in audio_mapping.items():
        if pretrained_key in pretrained_dict and fcn_key in fcn_state_dict:
            pretrained_param = pretrained_dict[pretrained_key]
            fcn_param = fcn_state_dict[fcn_key]
            
            # Check shape compatibility
            if pretrained_param.shape == fcn_param.shape:
                new_state_dict[fcn_key] = pretrained_param
                stats['audio_conv_loaded'] += 1
                stats['loaded_param_names'].append(f"{pretrained_key} → {fcn_key}")
                if verbose:
                    print(f"  ✓ {pretrained_key} → {fcn_key} {list(pretrained_param.shape)}")
            else:
                stats['audio_conv_skipped'] += 1
                stats['skipped_param_names'].append(f"{pretrained_key} (shape mismatch)")
                if verbose:
                    print(f"  ✗ {pretrained_key} SKIPPED (shape mismatch: "
                          f"{pretrained_param.shape} vs {fcn_param.shape})")
    
    for pretrained_key, fcn_key in video_mapping.items():
        if pretrained_key in pretrained_dict and fcn_key in fcn_state_dict:
            pretrained_param = pretrained_dict[pretrained_key]
            fcn_param = fcn_state_dict[fcn_key]
            
            # Check shape compatibility
            if pretrained_param.shape == fcn_param.shape:
                new_state_dict[fcn_key] = pretrained_param
                stats['video_conv_loaded'] += 1
                stats['loaded_param_names'].append(f"{pretrained_key} → {fcn_key}")
                if verbose:
                    print(f"  ✓ {pretrained_key} → {fcn_key} {list(pretrained_param.shape)}")
            else:
                stats['video_conv_skipped'] += 1
                stats['skipped_param_names'].append(f"{pretrained_key} (shape mismatch)")
                if verbose:
                    print(f"  ✗ {pretrained_key} SKIPPED (shape mismatch: "
                          f"{pretrained_param.shape} vs {fcn_param.shape})")
    
    # Count skipped FC layers
    for key in pretrained_dict.keys():
        if 'netfcaud' in key or 'netfclip' in key:
            stats['fc_layers_skipped'] += 1
    
    # Load the new state dict
    fcn_model.load_state_dict(new_state_dict, strict=False)
    
    if verbose:
        print("\n" + "="*80)
        print("TRANSFER LEARNING SUMMARY")
        print("="*80)
        print(f"Audio conv layers loaded:   {stats['audio_conv_loaded']}")
        print(f"Audio conv layers skipped:  {stats['audio_conv_skipped']}")
        print(f"Video conv layers loaded:   {stats['video_conv_loaded']}")
        print(f"Video conv layers skipped:  {stats['video_conv_skipped']}")
        print(f"FC layers skipped:          {stats['fc_layers_skipped']}")
        print("="*80)
    
    # Freeze loaded conv layers if requested
    if freeze_conv_layers:
        if verbose:
            print("\nFreezing loaded conv layers...")
        freeze_stats = freeze_pretrained_layers(fcn_model, verbose=verbose)
        stats.update(freeze_stats)
    
    if verbose:
        print_trainable_summary(fcn_model)
    
    return stats


def create_audio_mapping(pretrained_dict, fcn_state_dict, verbose=False):
    """
    Create mapping from original SyncNet audio layers to FCN audio encoder.
    
    Original: netcnnaud.0.weight, netcnnaud.0.bias, netcnnaud.1.weight, etc.
    FCN:      audio_encoder.conv_layers.0.weight, audio_encoder.conv_layers.0.bias, etc.
    """
    mapping = {}
    
    # Find all netcnnaud layers
    netcnnaud_keys = [k for k in pretrained_dict.keys() if k.startswith('netcnnaud.')]
    
    for key in netcnnaud_keys:
        # Extract layer number and parameter name
        # e.g., "netcnnaud.0.weight" → layer_idx=0, param_name="weight"
        parts = key.split('.')
        if len(parts) >= 3:
            layer_idx = parts[1]
            param_name = '.'.join(parts[2:])
            
            # Map to FCN audio encoder
            fcn_key = f"audio_encoder.conv_layers.{layer_idx}.{param_name}"
            
            # Check if this key exists in FCN model
            if fcn_key in fcn_state_dict:
                mapping[key] = fcn_key
    
    return mapping


def create_video_mapping(pretrained_dict, fcn_state_dict, verbose=False):
    """
    Create mapping from original SyncNet video layers to FCN video encoder.
    
    Original: netcnnlip.0.weight, netcnnlip.0.bias, netcnnlip.1.weight, etc.
    FCN:      video_encoder.conv_layers.0.weight, video_encoder.conv_layers.0.bias, etc.
    """
    mapping = {}
    
    # Find all netcnnlip layers
    netcnnlip_keys = [k for k in pretrained_dict.keys() if k.startswith('netcnnlip.')]
    
    for key in netcnnlip_keys:
        # Extract layer number and parameter name
        parts = key.split('.')
        if len(parts) >= 3:
            layer_idx = parts[1]
            param_name = '.'.join(parts[2:])
            
            # Map to FCN video encoder
            fcn_key = f"video_encoder.conv_layers.{layer_idx}.{param_name}"
            
            # Check if this key exists in FCN model
            if fcn_key in fcn_state_dict:
                mapping[key] = fcn_key
    
    return mapping


def freeze_pretrained_layers(fcn_model, verbose=True):
    """
    Freeze all conv layers in audio and video encoders.
    Keep channel_conv, attention, correlation, and predictor trainable.
    
    Args:
        fcn_model: SyncNetFCN model
        verbose: Print freezing details
        
    Returns:
        stats: Dictionary with freezing statistics
    """
    stats = {
        'frozen_params': 0,
        'trainable_params': 0,
        'frozen_layers': [],
        'trainable_layers': []
    }
    
    for name, param in fcn_model.named_parameters():
        # Freeze conv_layers in both encoders
        if 'audio_encoder.conv_layers' in name or 'video_encoder.conv_layers' in name:
            param.requires_grad = False
            stats['frozen_params'] += param.numel()
            stats['frozen_layers'].append(name)
        else:
            param.requires_grad = True
            stats['trainable_params'] += param.numel()
            stats['trainable_layers'].append(name)
    
    if verbose:
        print(f"  Frozen parameters:    {stats['frozen_params']:,}")
        print(f"  Trainable parameters: {stats['trainable_params']:,}")
    
    return stats


def unfreeze_all_layers(fcn_model, verbose=True):
    """
    Unfreeze all layers in the model for fine-tuning.
    
    Args:
        fcn_model: SyncNetFCN model
        verbose: Print details
    """
    total_params = 0
    for param in fcn_model.parameters():
        param.requires_grad = True
        total_params += param.numel()
    
    if verbose:
        print(f"All layers unfrozen. Total trainable parameters: {total_params:,}")


def print_trainable_summary(fcn_model):
    """
    Print a summary of trainable vs frozen parameters.
    """
    trainable_params = sum(p.numel() for p in fcn_model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in fcn_model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params
    
    print("\n" + "="*80)
    print("MODEL PARAMETER SUMMARY")
    print("="*80)
    print(f"Frozen parameters:     {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"Trainable parameters:  {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"Total parameters:      {total_params:,}")
    print("="*80)
    
    # List trainable layer groups
    print("\nTrainable layer groups:")
    trainable_groups = set()
    for name, param in fcn_model.named_parameters():
        if param.requires_grad:
            # Extract high-level group name
            group = name.split('.')[0] + '.' + name.split('.')[1] if len(name.split('.')) > 1 else name.split('.')[0]
            trainable_groups.add(group)
    
    for group in sorted(trainable_groups):
        print(f"  • {group}")
    
    print("\nFrozen layer groups:")
    frozen_groups = set()
    for name, param in fcn_model.named_parameters():
        if not param.requires_grad:
            group = name.split('.')[0] + '.' + name.split('.')[1] if len(name.split('.')) > 1 else name.split('.')[0]
            frozen_groups.add(group)
    
    for group in sorted(frozen_groups):
        print(f"  • {group}")
    print()


def get_optimizer_param_groups(fcn_model, pretrained_lr=1e-5, new_layers_lr=1e-3):
    """
    Create parameter groups for different learning rates.
    Pretrained conv layers get smaller LR, new layers get larger LR.
    
    Args:
        fcn_model: SyncNetFCN model
        pretrained_lr: Learning rate for pretrained conv layers
        new_layers_lr: Learning rate for new layers
        
    Returns:
        param_groups: List of parameter groups for optimizer
        
    Example usage:
        param_groups = get_optimizer_param_groups(model)
        optimizer = torch.optim.Adam(param_groups)
    """
    pretrained_params = []
    new_params = []
    
    for name, param in fcn_model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters
            
        if 'conv_layers' in name and ('audio_encoder' in name or 'video_encoder' in name):
            pretrained_params.append(param)
        else:
            new_params.append(param)
    
    param_groups = [
        {'params': pretrained_params, 'lr': pretrained_lr, 'name': 'pretrained_conv'},
        {'params': new_params, 'lr': new_layers_lr, 'name': 'new_layers'}
    ]
    
    print(f"Optimizer param groups created:")
    print(f"  Pretrained conv layers: {sum(p.numel() for p in pretrained_params):,} params @ lr={pretrained_lr}")
    print(f"  New layers:             {sum(p.numel() for p in new_params):,} params @ lr={new_layers_lr}")
    
    return param_groups


# Example usage and testing
if __name__ == "__main__":
    from SyncNetModel_FCN import SyncNetFCN, SyncNetFCN_WithAttention
    
    print("\n" + "="*80)
    print("TRANSFER LEARNING UTILITY - DEMO")
    print("="*80)
    
    # Create FCN model
    print("\nCreating FCN model...")
    fcn_model = SyncNetFCN(embedding_dim=512, max_offset=15)
    
    print("\nBefore transfer learning:")
    print_trainable_summary(fcn_model)
    
    # Simulate transfer learning (you would use actual checkpoint path)
    print("\n" + "="*80)
    print("To use transfer learning, call:")
    print("="*80)
    print("""
from transfer_learning_utils import load_pretrained_syncnet

# Load pretrained weights and freeze conv layers
stats = load_pretrained_syncnet(
    fcn_model=model,
    pretrained_syncnet_path='path/to/syncnet_v2.model',
    freeze_conv_layers=True,
    verbose=True
)

# Create optimizer with different learning rates
from torch.optim import Adam
param_groups = get_optimizer_param_groups(
    model, 
    pretrained_lr=1e-5,  # Small LR for pretrained layers
    new_layers_lr=1e-3   # Larger LR for new layers
)
optimizer = Adam(param_groups)
    """)
    
    print("\n" + "="*80)
    print("For fine-tuning all layers later:")
    print("="*80)
    print("""
from transfer_learning_utils import unfreeze_all_layers

# Unfreeze all layers for fine-tuning
unfreeze_all_layers(model, verbose=True)

# Use smaller learning rate for fine-tuning
optimizer = Adam(model.parameters(), lr=1e-4)
    """)
