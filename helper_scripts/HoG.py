import os
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from skimage.feature import hog
from tqdm import tqdm

input_dir = r"dataset_1\GRK-papyri\All\All"
model_path = r"weights\best.pt"
output_pickle = r"hog_features.pkl"
os.makedirs(os.path.dirname(output_pickle), exist_ok=True)

patch_size = 256
stride = 128
conf_thresh = 0.3
resize_shape = (64, 64)

# HoG parameters
hog_params = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    transform_sqrt=True,
    feature_vector=True
)

model = YOLO(model_path)
features, labels, sample_ids = [], [], []

sample_id = 0
for root, _, files in os.walk(input_dir):
    for file in tqdm(files, desc="Processing images"):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(root, file)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        h, w = gray.shape

        base_name = os.path.splitext(file)[0]

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w-patch_size+1, stride):
                patch = rgb[y:y+patch_size, x:x+patch_size]
                # use model to detect
                result = model(patch, conf=conf_thresh)[0]
                boxes = result.boxes
                if boxes is not None and boxes.shape[0] > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cls_id = int(box.cls[0].item())
                        x1 = xyxy[0]+x
                        y1 = xyxy[1]+y
                        x2 = xyxy[2]+x
                        y2 = xyxy[3]+y

                        char_crop = gray[y1:y2, x1:x2]

                        if char_crop.shape[0] < 8 or char_crop.shape[1] < 8:
                            continue
                        # hog feature extraction
                        char_crop_resized = cv2.resize(char_crop, resize_shape)
                        feature = hog(char_crop_resized, **hog_params)
                        features.append(feature)
                        labels.append(cls_id)
                        sample_ids.append(sample_id)
        sample_id += 1

with open(output_pickle, "wb") as f:
    pickle.dump({"features": np.array(features),"labels": np.array(labels),"sample_ids": np.array(sample_ids)
    }, f)

print(f"Save{len(features)} HoG features to {output_pickle}")
