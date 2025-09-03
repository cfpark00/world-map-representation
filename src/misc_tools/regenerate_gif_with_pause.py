#!/usr/bin/env python3
"""
Regenerate the world map evolution GIF with a pause at the final frame.
"""

from pathlib import Path
from PIL import Image

def add_pause_to_gif(input_gif_path, output_gif_path=None, final_frame_duration=3000):
    """
    Modify a GIF to add a pause on the final frame.
    
    Args:
        input_gif_path: Path to the input GIF
        output_gif_path: Path for output GIF (if None, overwrites input)
        final_frame_duration: Duration for final frame in milliseconds (default 1500ms)
    """
    input_path = Path(input_gif_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input GIF not found: {input_path}")
    
    if output_gif_path is None:
        output_gif_path = input_path
    else:
        output_gif_path = Path(output_gif_path)
    
    # Open the GIF and extract all frames
    img = Image.open(input_path)
    frames = []
    durations = []
    
    try:
        while True:
            # Get frame duration (default to 500ms if not specified)
            duration = img.info.get('duration', 500)
            durations.append(duration)
            
            # Copy and save the frame
            frames.append(img.copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass  # End of GIF
    
    if not frames:
        raise ValueError("No frames found in GIF")
    
    # Modify the duration of the last frame
    durations[-1] = final_frame_duration
    
    print(f"Found {len(frames)} frames")
    print(f"Original durations: {durations[:5]}... (showing first 5)")
    print(f"Modified last frame duration: {durations[-1]}ms")
    
    # Save the modified GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )
    
    print(f"Saved modified GIF to: {output_gif_path}")

def main():
    # Path to the GIF in the blog post directory
    blog_gif = Path('/n/home12/cfpark00/WM_1/reports/emergent-geographic-representations/world_map_evolution.gif')
    
    # Also check the analysis directory (source)
    analysis_gif = Path('/n/home12/cfpark00/WM_1/outputs/experiments/dist_100k_1M_20epochs/analysis/layers3_4_probe5000_train3000/world_map_evolution.gif')
    
    if blog_gif.exists():
        print(f"Processing blog GIF: {blog_gif}")
        add_pause_to_gif(blog_gif, final_frame_duration=3000)  # 2.5s extra pause
    else:
        print(f"Blog GIF not found: {blog_gif}")
    
    if analysis_gif.exists():
        print(f"\nProcessing analysis GIF: {analysis_gif}")
        add_pause_to_gif(analysis_gif, final_frame_duration=3000)  # 2.5s extra pause
    else:
        print(f"Analysis GIF not found: {analysis_gif}")

if __name__ == "__main__":
    main()