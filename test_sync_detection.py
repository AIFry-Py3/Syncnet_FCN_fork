#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Test Script for StreamSyncFCN

Run this to detect sync offset in your videos.

Usage:
    python test_sync_detection.py --video path/to/video.mp4
    
With pretrained weights:
    python test_sync_detection.py --video video.mp4 --model data/syncnet_v2.model
    
For HLS:
    python test_sync_detection.py --hls http://example.com/stream.m3u8 --model data/syncnet_v2.model
"""

import argparse
from SyncNetModel_FCN import StreamSyncFCN

def main():
    parser = argparse.ArgumentParser(description='Detect audio-video sync offset')
    parser.add_argument('--video', type=str, help='Path to video file (MP4, AVI, MOV, etc.)')
    parser.add_argument('--hls', type=str, help='HLS stream URL (.m3u8)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pretrained SyncNet model file (optional)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration to capture for HLS (seconds, default: 10)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.video and not args.hls:
        print("Error: Please provide either --video or --hls")
        parser.print_help()
        return
    
    print("="*80)
    print("StreamSyncFCN - Sync Offset Detection")
    print("="*80)
    
    # Create model
    print("\nInitializing model...")
    model = StreamSyncFCN(
        pretrained_syncnet_path=args.model,
        auto_load_pretrained=(args.model is not None)
    )
    
    if args.model:
        print(f"✓ Loaded pretrained weights from: {args.model}")
    else:
        print("⚠ Using random initialization (no pretrained weights)")
    
    # Process video or HLS
    try:
        if args.video:
            print(f"\nProcessing video file: {args.video}")
            print("-"*80)
            offset, confidence = model.process_video_file(args.video, verbose=True)
            
        elif args.hls:
            print(f"\nProcessing HLS stream: {args.hls}")
            print(f"Capturing {args.duration} seconds...")
            print("-"*80)
            offset, confidence = model.process_hls_stream(
                args.hls,
                segment_duration=args.duration,
                verbose=True
            )
        
        # Interpret results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Detected offset: {offset:.2f} frames")
        print(f"Confidence:      {confidence:.3f}")
        
        if offset > 1:
            print(f"\n→ Audio is {offset:.2f} frames AHEAD of video")
            print(f"   (Audio plays {offset/25:.3f} seconds earlier)")
        elif offset < -1:
            print(f"\n→ Audio is {abs(offset):.2f} frames BEHIND video")
            print(f"   (Audio plays {abs(offset)/25:.3f} seconds later)")
        else:
            print(f"\n→ Audio and video are IN SYNC ✓")
        
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found")
        print(f"  {e}")
    except Exception as e:
        print(f"\n✗ Error during processing:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
