import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, stride=1, start_frame=0, end_frame=None, 
                   image_format='jpg', quality=95):
    """
    Extract frames from a video with specified stride.
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save extracted frames
        stride (int): Extract every Nth frame (default: 1 = every frame)
        start_frame (int): Frame number to start extraction (default: 0)
        end_frame (int): Frame number to end extraction (default: None = end of video)
        image_format (str): Output image format ('jpg', 'png', 'bmp')
        quality (int): JPEG quality (0-100, only for jpg format)
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames
    
    # Validate frame range
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)
    
    print(f"Extracting frames {start_frame} to {end_frame} with stride {stride}")
    
    # Set encoding parameters
    if image_format.lower() == 'jpg':
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = '.jpg'
    elif image_format.lower() == 'png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        ext = '.png'
    else:
        encode_params = []
        ext = f'.{image_format.lower()}'
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while True:
        ret, frame = cap.read()
        
        if not ret or (start_frame + frame_count) >= end_frame:
            break
        
        # Check if this frame should be saved (based on stride)
        if frame_count % stride == 0:
            frame_filename = f"frame_{start_frame + frame_count:06d}{ext}"
            frame_path = output_path / frame_filename
            
            # Save frame
            success = cv2.imwrite(str(frame_path), frame, encode_params)
            
            if success:
                saved_count += 1
                if saved_count % 100 == 0:  # Progress update every 100 frames
                    print(f"Saved {saved_count} frames...")
            else:
                print(f"Warning: Failed to save frame {frame_filename}")
        
        frame_count += 1
    
    cap.release()
    print(f"Extraction complete!")
    print(f"Saved {saved_count} frames to {output_dir}")
    
    return saved_count

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video with specified stride")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_dir", help="Output directory for extracted frames")
    parser.add_argument("--stride", type=int, default=1, 
                       help="Extract every Nth frame (default: 1)")
    parser.add_argument("--start", type=int, default=0,
                       help="Starting frame number (default: 0)")
    parser.add_argument("--end", type=int, default=None,
                       help="Ending frame number (default: end of video)")
    parser.add_argument("--format", choices=['jpg', 'png', 'bmp'], default='jpg',
                       help="Output image format (default: jpg)")
    parser.add_argument("--quality", type=int, default=95,
                       help="JPEG quality 0-100 (default: 95)")
    
    args = parser.parse_args()
    
    try:
        extract_frames(
            video_path=args.video_path,
            output_dir=args.output_dir,
            stride=args.stride,
            start_frame=args.start,
            end_frame=args.end,
            image_format=args.format,
            quality=args.quality
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Example usage:
if __name__ == "__main__" and False:  # Set to True to run examples
    # Extract every 10th frame
    extract_frames("input_video.mp4", "frames_output", stride=10)
    
    # Extract frames 100-500, every 5th frame
    extract_frames("input_video.mp4", "frames_subset", 
                   stride=5, start_frame=100, end_frame=500)
    
    # Extract as PNG format
    extract_frames("input_video.mp4", "frames_png", 
                   stride=1, image_format='png')