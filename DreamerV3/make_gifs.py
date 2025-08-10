import os
import glob
from PIL import Image

def make_gif(frame_folder, output_gif, pattern="*.png", duration=100):
    """Create a GIF from PNG frames in a folder."""
    # Get sorted list of PNG files
    frames = sorted(glob.glob(os.path.join(frame_folder, pattern)))
    if not frames:
        print(f"No PNG files found in {frame_folder}")
        return False

    # Open images
    images = [Image.open(f) for f in frames]
    # Save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved to {output_gif} ({len(frames)} frames)")
    return True

def make_gifs_from_inference_frames(base_folder="inference_frames", pattern="*.png", duration=100):
    """Automatically create GIFs for each subfolder in inference_frames."""
    if not os.path.exists(base_folder):
        print(f"Base folder {base_folder} does not exist")
        return

    # Get all subdirectories in inference_frames
    subfolders = [f for f in os.listdir(base_folder) 
                  if os.path.isdir(os.path.join(base_folder, f))]
    
    if not subfolders:
        print(f"No subfolders found in {base_folder}")
        return

    print(f"Found {len(subfolders)} subfolders in {base_folder}")
    
    # Create GIF for each subfolder
    for subfolder in sorted(subfolders):
        frame_folder = os.path.join(base_folder, subfolder)
        output_gif = os.path.join(base_folder, f"{subfolder}.gif")
        
        print(f"Processing {subfolder}...")
        success = make_gif(frame_folder, output_gif, pattern, duration)
        
        if not success:
            print(f"  Warning: No frames found in {subfolder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Make GIFs from PNG frames.")
    parser.add_argument("--base_folder", default="inference_frames", 
                       help="Base folder containing subfolders with frames (default: inference_frames)")
    parser.add_argument("--single_folder", help="Process only a single folder instead of all subfolders")
    parser.add_argument("--output_gif", help="Output GIF file path (required if using --single_folder)")
    parser.add_argument("--pattern", default="*.png", help="Glob pattern for frames (default: *.png)")
    parser.add_argument("--duration", type=int, default=100, help="Duration per frame in ms (default: 100)")
    
    args = parser.parse_args()

    if args.single_folder:
        if not args.output_gif:
            print("Error: --output_gif is required when using --single_folder")
            exit(1)
        make_gif(args.single_folder, args.output_gif, args.pattern, args.duration)
    else:
        make_gifs_from_inference_frames(args.base_folder, args.pattern, args.duration)