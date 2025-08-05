import json, os, cv2
from pathlib import Path
import tqdm

images_dir = Path("./papytwin/HomerCompTraining/")
predictions_path = Path("./papytwin/predictions/CV_2Twin_res50_box96_train/predictions.json")
train_json = Path("./papytwin/HomerCompTraining/HomerCompTraining.json")
output_dir = Path("glyph_crops/train")
output_dir.mkdir(parents=True, exist_ok=True)

# Load dataset metadata
with open(train_json) as f:
    train_data = json.load(f)

id_to_filename = {img["id"]: img["file_name"] for img in train_data["images"]}

# Load predictions
with open(predictions_path) as f:
    predictions = json.load(f)
    predictions = predictions["annotations"]

# Counters
invalid_files = 0
invalid_bboxes = 0
saved_crops = 0

# Loop through predictions
for i, pred in tqdm.tqdm(enumerate(predictions), total=len(predictions)):
    img_id = pred["image_id"]
    bbox = list(map(int, pred["bbox"]))
    x, y, w, h = bbox
    score = pred["score"]
    cat = pred["category_id"]

    if score < 0.7:
        continue

    filename = id_to_filename.get(img_id)
    if filename is None:
        invalid_files += 1
        continue

    image_path = images_dir / filename
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[Warning] Could not load image: {image_path}")
            invalid_files += 1
            continue

        h_img, w_img = img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        if x2 <= x or y2 <= y:
            print(f"[Warning] Invalid bounding box {bbox} for image {image_path}")
            invalid_bboxes += 1
            continue

        crop = img[y:y2, x:x2]
        if crop.size == 0:
            print(f"[Warning] Empty crop for bbox {bbox} in image {image_path}")
            invalid_bboxes += 1
            continue

        crop_path = output_dir / f"{img_id}_{i}_cat{cat}.jpg"
        cv2.imwrite(str(crop_path), crop)
        saved_crops += 1

    except Exception as ex:
        print(f"[Error] Exception while processing image {image_path}: {ex}")
        invalid_files += 1
        continue

# Final summary
print("\nProcessing complete.")
print(f"Total crops saved      : {saved_crops}")
print(f"Invalid/missing images : {invalid_files}")
print(f"Invalid bounding boxes : {invalid_bboxes}")
