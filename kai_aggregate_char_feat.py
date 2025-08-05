# aggregate_character_features.py (FINAL CORRECTED VERSION)
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# --- CONFIG ---
TARGET_CHAR_NAME = "kai"
input_embeddings_path = f"{TARGET_CHAR_NAME}_embeddings.npz"
output_descriptors_path = f"page_descriptors_{TARGET_CHAR_NAME}.npz"
json_path = Path("papytwin/HomerCompTraining/HomerCompTraining.json")
# --- END CONFIG ---

# --- Step 1: Load metadata ---
print("Loading metadata...")
with open(json_path) as f:
    data = json.load(f)
image_id_to_file_name = {
    img["id"]: img["file_name"]  # Keep the full path here for now
    for img in data["images"]
}

# --- Step 2: Load character embeddings ---
print(f"Loading character embeddings from: {input_embeddings_path}")
data = np.load(input_embeddings_path, allow_pickle=True)
embeddings = data["embeddings"]
paths = data["paths"]
print(f"Loaded {len(embeddings)} embeddings for character '{TARGET_CHAR_NAME}'.")

# --- Step 3: Aggregate embeddings by page ---
page_features = defaultdict(list)
print("Grouping embeddings by document page...")
for emb, path in tqdm(zip(embeddings, paths), total=len(paths), desc="Aggregating"):
    try:
        image_id = int(Path(path).stem.split("_")[0])
        doc_full_path = image_id_to_file_name.get(image_id)

        if doc_full_path:
            # ✅ CORRECTED LOGIC: Get just the filename (basename) of the path
            doc_basename = Path(doc_full_path).name
            page_features[doc_basename].append(emb)
    except Exception:
        continue

# --- Step 4: Calculate Mean Descriptor for each Page ---
page_descriptors = []
page_ids = []
print("Calculating mean page descriptors...")
for doc_basename, embs in tqdm(page_features.items(), desc="Averaging"):
    if embs:
        mean_descriptor = np.mean(embs, axis=0)
        page_descriptors.append(mean_descriptor)
        page_ids.append(doc_basename)

# --- DEBUGGING BLOCK ---
print("\n--- DEBUG: First 5 Page IDs being saved ---")
for i in range(min(5, len(page_ids))):
    print(f"  '{page_ids[i]}'")
print("------------------------------------------\n")

print(f"Created {len(page_descriptors)} page-level descriptors.")

# --- Step 5: Save the final page descriptors ---
np.savez_compressed(
    output_descriptors_path,
    descriptors=np.array(page_descriptors),
    page_ids=np.array(page_ids)
)
print(f"✅ Success! Saved aggregated page descriptors to '{output_descriptors_path}'")