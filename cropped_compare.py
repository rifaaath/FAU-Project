from PIL import Image
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm

def summarize_image_folder(folder):
    widths, heights, aspect_ratios, mean_pixels, file_sizes = [], [], [], [], []
    folder = Path(folder)
    image_files = list(folder.glob("*.jpg"))

    for img_path in tqdm(image_files, total=len(image_files)):
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            mean_pixels.append(np.array(img).mean())
            file_sizes.append(os.path.getsize(img_path) / 1024)  # in KB
        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")

    return {
        "count": len(image_files),
        "avg_width": np.mean(widths),
        "std_width": np.std(widths),
        "avg_height": np.mean(heights),
        "std_height": np.std(heights),
        "avg_aspect_ratio": np.mean(aspect_ratios),
        "avg_pixel_value": np.mean(mean_pixels),
        "avg_file_size_kb": np.mean(file_sizes)
    }

# Paths to your datasets
opencv_dir = "glyph_crops/train"
kornia_dir = "glyph_crops/batch_cropper"

opencv_stats = summarize_image_folder(opencv_dir)
kornia_stats = summarize_image_folder(kornia_dir)

print("OpenCV Crops:")
for k, v in opencv_stats.items():
    print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

print("\nKornia Crops:")
for k, v in kornia_stats.items():
    print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")