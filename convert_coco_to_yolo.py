# convert_coco_to_yolo.py (Corrected for Directory Structure)
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import shutil

# --- Config ---
json_path = Path("papytwin/HomerCompTraining/HomerCompTraining.json")

# --- ✅ CORRECTED SOURCE IMAGE DIRECTORY ---
# This is the base directory where the image paths from the JSON file are relative to.
images_source_dir = Path("papytwin/HomerCompTraining/")

output_dir = Path("datasets/yolo_glyphs")
# --- End Config ---

print(f"Loading COCO data from {json_path}...")
with open(json_path) as f:
    coco_data = json.load(f)

# Create YOLO directory structure
images_train_dir = output_dir / "images" / "train"
labels_train_dir = output_dir / "labels" / "train"
images_train_dir.mkdir(parents=True, exist_ok=True)
labels_train_dir.mkdir(parents=True, exist_ok=True)

# Create a map of image_id to image info
image_info_map = {img['id']: img for img in coco_data['images']}

# Group annotations by image_id
annotations_by_image = defaultdict(list)
for ann in coco_data['annotations']:
    annotations_by_image[ann['image_id']].append(ann)

print("Converting COCO annotations to YOLO format...")
files_copied = 0
files_skipped = 0

# Process each image and its annotations
for image_id, annotations in tqdm(annotations_by_image.items(), desc="Processing images"):
    if image_id not in image_info_map:
        continue

    image_info = image_info_map[image_id]
    img_w = image_info['width']
    img_h = image_info['height']
    # The filename from the JSON already has the subdirectory structure (homer2/txt107/...)
    relative_img_path = Path(image_info['file_name'])

    # --- ✅ CORRECTED PATH LOGIC ---
    # Construct the full source path correctly
    source_image_path = images_source_dir / relative_img_path
    # The destination filename is just the name part, not the whole path
    dest_image_path = images_train_dir / relative_img_path.name

    # --- Copy image file to the new flat location ---
    if source_image_path.exists():
        shutil.copy(source_image_path, dest_image_path)
        files_copied += 1
    else:
        # This will help debug if paths are still wrong
        # print(f"Warning: Source image not found: {source_image_path}")
        files_skipped += 1
        continue

    # --- Create the YOLO label file ---
    # The label filename must match the new, flat image filename
    yolo_label_path = labels_train_dir / f"{dest_image_path.stem}.txt"
    with open(yolo_label_path, 'w') as f_label:
        for ann in annotations:
            cat_id = ann['category_id']
            x, y, w, h = ann['bbox']

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # YOLO class IDs must be 0-indexed.
            yolo_class_id = cat_id - 1

            f_label.write(f"{yolo_class_id} {x_center} {y_center} {norm_w} {norm_h}\n")

print(f"\n✅ Conversion complete. YOLO dataset created in '{output_dir.resolve()}'")
print(f"   - Images copied: {files_copied}")
print(f"   - Images skipped (not found): {files_skipped}")