import json, os, cv2
import torch
import kornia as K
from pathlib import Path
import tqdm
from collections import defaultdict

# Paths
images_dir = Path("./papytwin/HomerCompTraining/")
# predictions_path = Path("./papytwin/predictions/CV_2Twin_res50_box96_train/predictions.json")
predictions_path = Path("kai_boxes.json")
train_json = Path("./papytwin/HomerCompTraining/HomerCompTraining.json")
output_dir = Path("glyph_crops/kai")
output_dir.mkdir(parents=True, exist_ok=True)

# Load dataset metadata
with open(train_json) as f:
    train_data = json.load(f)
id_to_filename = {img["id"]: img["file_name"] for img in train_data["images"]}

# Load predictions and group by image
with open(predictions_path) as f:
    predictions = json.load(f)["annotations"]

pred_by_image = defaultdict(list)
for pred in predictions:
    if pred["score"] >= 0.7:
        pred_by_image[pred["image_id"]].append(pred)

# Counters
invalid_files = 0
invalid_bboxes = 0
saved_crops = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Process per image
for img_id, preds in tqdm.tqdm(pred_by_image.items(), desc="Processing images"):
    filename = id_to_filename.get(img_id)
    if filename is None:
        invalid_files += 1
        continue

    image_path = images_dir / filename
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"[Warning] Could not load image: {image_path}")
        invalid_files += 1
        continue

    h_img, w_img = img_bgr.shape[:2]

    # Convert to tensor and normalize
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = K.image_to_tensor(img_rgb, keepdim=False).float() / 255.0  # [3, H, W]
    img_tensor = img_tensor = img_tensor.to(device)  # [3, H, W]

    boxes = []
    box_dims = []
    metadata = []

    for i, pred in enumerate(preds):
        x, y, w, h = map(int, pred["bbox"])
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        if x2 <= x1 or y2 <= y1:
            invalid_bboxes += 1
            continue

        # Normalized coordinates
        abs_box = [x1, y1, x2, y2]  # ⚠️ Remove normalization!
        boxes.append(abs_box)
        box_dims.append((y2 - y1, x2 - x1))
        metadata.append((img_id, i, pred["category_id"]))

    if not boxes:
        continue

    try:
        # Batch crop all glyphs
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)
        crops = []
        for box, size in zip(boxes_tensor, box_dims):
            x1, y1, x2, y2 = box.tolist()
            polygon = torch.tensor([[
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ]], dtype=torch.float32).to(device)  # shape: [1, 4, 2]

            single_crop = K.geometry.transform.crop_and_resize(
                img_tensor,
                boxes=polygon,
                size=size,
                mode="bilinear",
                align_corners=True
            )

            crops.append(single_crop.squeeze(0))

        # Save each crop
        for crop_tensor, (img_id, i, cat) in zip(crops, metadata):
            crop_img = (K.tensor_to_image(crop_tensor.cpu()) * 255).clip(0, 255).astype("uint8")
            crop_path = output_dir / f"{img_id}_{i}_cat{cat}.jpg"
            cv2.imwrite(str(crop_path), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            saved_crops += 1

    except Exception as ex:
        print(f"[Error] Failed while processing {image_path}: {ex}")
        invalid_files += 1
        continue

# Summary
print("\nProcessing complete.")
print(f"Total crops saved      : {saved_crops}")
print(f"Invalid/missing images : {invalid_files}")
print(f"Invalid bounding boxes : {invalid_bboxes}")
