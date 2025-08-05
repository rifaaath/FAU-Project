import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- CONFIG ---
TARGET_CHAR_NAME = "kai"
input_descriptors_path = f"page_descriptors_{TARGET_CHAR_NAME}.npz"
excel_path = "papytwin/1.CompetitionOverview.xlsx"
# --- END CONFIG ---

# --- Load Page Descriptors and Page IDs ---
print(f"Loading page descriptors from: {input_descriptors_path}")
data = np.load(input_descriptors_path, allow_pickle=True)
X_pages = data["descriptors"]
page_ids = data["page_ids"]  # Contains full filenames like 'p_18177_r_3_001.jpg'

# --- Load the mapping from Page ID to Writer ID ---
print(f"Loading writer metadata from: {excel_path}")
df_meta = pd.read_excel(excel_path)

# 1. Normalize the 'Image Name' column from the Excel file to get clean stems.
df_meta["Image Name Stem"] = df_meta["Image Name"].astype(str).str.strip().str.lower()
# 2. Create the lookup dictionary using the STEM as the key.
page_stem_to_writer = {
    stem: f"TM_{tm}"
    for stem, tm in zip(df_meta["Image Name Stem"], df_meta["TM with READ item name in ()"])
}

# --- Get Writer ID for each Page Descriptor ---
page_writer_ids = []
for pid in page_ids:
    # 3. For each page_id from your .npz file, get its stem and normalize it.
    stem_to_lookup = Path(pid).stem.lower().strip()
    # 4. Use this stem to look up the writer ID.
    writer_id = page_stem_to_writer.get(stem_to_lookup)
    page_writer_ids.append(writer_id)

page_writer_ids = np.array(page_writer_ids)


valid_mask = (page_writer_ids != None)
X_pages = X_pages[valid_mask]
page_writer_ids = page_writer_ids[valid_mask]

if len(X_pages) == 0:
    print(
        "\nError: No pages could be mapped to a writer.")
    exit()

print(f"Successfully mapped {len(X_pages)} pages to their writers.")

unique_writers = np.unique(page_writer_ids)
np.random.seed(42)
np.random.shuffle(unique_writers)
test_writer_ids = unique_writers[int(0.8 * len(unique_writers)):]
print(f"Evaluating on {len(test_writer_ids)} unseen writers.")

test_mask = np.isin(page_writer_ids, test_writer_ids)
X_test = X_pages[test_mask]
writer_ids_test = page_writer_ids[test_mask]

print("Building writer prototypes..")
writer_pages = defaultdict(list)
for page_desc, wid in zip(X_test, writer_ids_test):
    writer_pages[wid].append(page_desc)

gallery_writers = list(writer_pages.keys())
gallery_vectors = np.array([np.mean(descs, axis=0) for descs in writer_pages.values()])
gallery_writer_map = {writer: i for i, writer in enumerate(gallery_writers)}

print("Evaluating Top-1 retrieval accuracy...")
correct_predictions = 0
total_queries = 0

for query_wid, query_descs in tqdm(writer_pages.items(), desc="Evaluating writers"):
    if len(query_descs) < 2:
        continue

    for i, query_desc in enumerate(query_descs):
        temp_gallery_vectors = gallery_vectors.copy()
        other_descs = np.delete(query_descs, i, axis=0)
        writer_idx_in_gallery = gallery_writer_map[query_wid]
        temp_gallery_vectors[writer_idx_in_gallery] = np.mean(other_descs, axis=0)

        similarities = cosine_similarity(query_desc.reshape(1, -1), temp_gallery_vectors)[0]
        predicted_idx = np.argmax(similarities)
        predicted_writer = gallery_writers[predicted_idx]

        if predicted_writer == query_wid:
            correct_predictions += 1
        total_queries += 1

retrieval_accuracy = (correct_predictions / total_queries) * 100 if total_queries > 0 else 0
print(f"\n✅ Top-1 KaiRacters-Style Retrieval Accuracy (on multi-page writers): {retrieval_accuracy:.2f}%")