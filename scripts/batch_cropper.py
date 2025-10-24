import json, os, cv2
import torch
import kornia as K
from pathlib import Path
import tqdm
from collections import defaultdict
import argparse

def run_cropping(args):
    images_dir = Path(args.images_dir)
    predictions_path = Path(args.predictions_path)
    train_json = Path(args.train_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and predictions...")
    with open(train_json) as f:
        train_data = json.load(f)
    id_to_full_path = {img["id"]: img["file_name"] for img in train_data["images"]}
    filename_to_id = {Path(img["file_name"]).name: img["id"] for img in train_data["images"]}

    with open(predictions_path) as f:
        predictions_by_filename = json.load(f)

    pred_by_image_id = defaultdict(list)
    skipped_files = 0
    for filename, preds in predictions_by_filename.items():
        sanitized_filename = Path(filename).with_suffix('.jpg').name
        image_id = filename_to_id.get(filename) or filename_to_id.get(sanitized_filename)
        if image_id is not None:
            for p in preds:
                p['image_id'] = image_id
            pred_by_image_id[image_id].extend(preds)
        else:
            skipped_files += 1
    if skipped_files > 0:
        print(f"Warning: Could not find metadata for {skipped_files} image files. They will be skipped.")

    invalid_files, invalid_bboxes, saved_crops = 0, 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for img_id, preds in tqdm.tqdm(pred_by_image_id.items(), desc="Processing images"):
        try:
            relative_filename = id_to_full_path.get(img_id)
            if not relative_filename: continue
            image_path = images_dir / relative_filename
            if not image_path.exists(): continue
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                print(f"[Warning] Could not load image: {image_path}")
                invalid_files += 1
                continue
            h_img, w_img = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = K.image_to_tensor(img_rgb, keepdim=False).float() / 255.0
            img_tensor = img_tensor.to(device)
            boxes, box_dims, metadata = [], [], []
            for i, pred in enumerate(preds):
                x, y, w, h = map(int, pred["bbox"])
                x1, y1, x2, y2 = max(0, x), max(0, y), min(w_img, x + w), min(h_img, y + h)
                if x2 <= x1 or y2 <= y1:
                    invalid_bboxes += 1
                    continue
                boxes.append([x1, y1, x2, y2])
                box_dims.append((y2 - y1, x2 - x1))
                metadata.append((img_id, i, pred["category_id"]))
            if not boxes: continue
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)
            crops = []
            batch_img_tensor = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
            if batch_img_tensor.dim() != 4:
                print(f"Skipping image {image_path} due to unexpected tensor dimension: {img_tensor.dim()}")
                continue
            for box, size in zip(boxes_tensor, box_dims):
                x1, y1, x2, y2 = box.tolist()
                polygon = torch.tensor([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=torch.float32).to(device)
                single_crop = K.geometry.transform.crop_and_resize(
                    batch_img_tensor, boxes=polygon, size=size, mode="bilinear", align_corners=True
                )
                crops.append(single_crop.squeeze(0))
            for crop_tensor, (img_id, i, cat) in zip(crops, metadata):
                crop_img = (K.tensor_to_image(crop_tensor.cpu()) * 255).clip(0, 255).astype("uint8")
                crop_path = output_dir / f"{img_id}_{i}_cat{cat}.jpg"
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
                saved_crops += 1
        except Exception as ex:
            print(f"[Error] Failed while processing image ID {img_id}. Details: {ex}")
            invalid_files += 1
            continue
    print("\nProcessing complete.")
    print(f"Total crops saved: {saved_crops}")
    print(f"Skipped unregistered files: {skipped_files}")
    print(f"Invalid/failed images: {invalid_files}")
    print(f"Invalid bounding boxes: {invalid_bboxes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop glyphs from full images based on prediction JSON.")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    args = parser.parse_args()
    run_cropping(args)