#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
StreamSyncFCN Usage Examples

Shows how to use the StreamSyncFCN model with:
1. Automatic transfer learning
2. Video file processing
3. HLS stream processing
"""

from SyncNetModel_FCN import StreamSyncFCN

print("="*80)
print("StreamSyncFCN - Complete Example")
print("="*80)

# Example 1: Create model WITH automatic transfer learning
print("\n" + "="*80)
print("Example 1: Auto-load pretrained SyncNet weights")
print("="*80)

model = StreamSyncFCN(
    pretrained_syncnet_path='data/syncnet_v2.model',  # Path to original SyncNet
    auto_load_pretrained=True,  # Automatically load and freeze conv layers
    window_size=25,
    stride=5,
    buffer_size=50
)

# The model is now ready with pretrained conv layers!
# Conv layers are frozen, new layers are trainable

# Example 2: Process a video file
print("\n" + "="*80)
print("Example 2: Process video file")
print("="*80)

# Just give it a file path!
# offset, confidence = model.process_video_file('path/to/your/video.mp4')

print(f"\nUsage:")
print(f"  offset, confidence = model.process_video_file('video.mp4')")
print(f"\nSupported formats: MP4, AVI, MOV, MKV, WebM, etc.")

# Example 3: Process HLS stream
print("\n" + "="*80)
print("Example 3: Process HLS stream (.m3u8)")
print("="*80)

print(f"\nUsage:")
print(f"  offset, conf = model.process_hls_stream('http://url/stream.m3u8')")

# Example 4: Model without pretrained weights
print("\n" + "="*80)
print("Example 4: Model without pretrained weights (random init)")
print("="*80)

model_random = StreamSyncFCN(
    auto_load_pretrained=False  # Don't load pretrained weights
)

print("Created model with random initialization (for training from scratch)")

print("\n" + "="*80)
print("SUMMARY: Supported Input Formats")
print("="*80)
print("""
Video Files:
  ✓ MP4  - model.process_video_file('video.mp4')
  ✓ AVI  - model.process_video_file('video.avi')
  ✓ MOV  - model.process_video_file('video.mov')
  ✓ MKV  - model.process_video_file('video.mkv')
  ✓ WebM - model.process_video_file('video.webm')

Streaming:
  ✓ HLS (.m3u8) - model.process_hls_stream('http://url/stream.m3u8')

Features:
  ✓ Automatic transfer learning from SyncNetModel.py
  ✓ Conv layers frozen, new layers trainable
  ✓ Sliding window processing
  ✓ Temporal buffering and smoothing
  ✓ Built-in preprocessing (MFCC + frame extraction)
""")

print("="*80)
print("Ready to use! 🚀")
print("="*80)
