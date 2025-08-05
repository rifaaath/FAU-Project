# evaluate_vlad_with_map.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# --- Config ---
# The VLAD descriptors you just created
vlad_descriptors_path = "page_descriptors_class_vlad.npz"
# The full manifest is needed to map doc_ids back to writer_ids
full_manifest_path = "glyph_manifest_full.csv"
# --- End Config ---

# --- Step 1: Load VLAD descriptors and create doc -> descriptor mapping ---
print(f"Loading VLAD page descriptors from: {vlad_descriptors_path}")
data = np.load(vlad_descriptors_path, allow_pickle=True)
descriptors = data["descriptors"]
doc_ids_from_vlad = data["doc_ids"]
doc_to_descriptor = {doc_id: desc for doc_id, desc in zip(doc_ids_from_vlad, descriptors)}

# --- Step 2: Create the doc -> writer mapping from the full manifest ---
print(f"Loading full manifest from: {full_manifest_path}")
df_full = pd.read_csv(full_manifest_path)


def get_doc_id(path):
    parts = Path(path).stem.split("_")
    if len(parts) > 2: return f"{parts[0]}_{parts[1]}"
    return Path(path).stem


df_full['doc_id'] = df_full['path'].apply(get_doc_id)

# Create a unique mapping from each document to its writer
doc_to_writer = df_full.drop_duplicates(subset='doc_id').set_index('doc_id')['tm_id'].to_dict()

# --- Step 3: Create the final test set for evaluation ---
# We will use the writers from the VLAD output, which should be all of them
all_doc_ids_in_vlad = list(doc_to_descriptor.keys())
all_writers_in_vlad = list(set([doc_to_writer.get(did) for did in all_doc_ids_in_vlad if doc_to_writer.get(did)]))
np.random.seed(42)
np.random.shuffle(all_writers_in_vlad)

test_writers = all_writers_in_vlad[int(0.8 * len(all_writers_in_vlad)):]
print(f"Evaluating on {len(test_writers)} unseen writers.")

# --- Build the gallery of writer prototypes from the TEST set ---
print("Building writer prototypes for the test set...")
writer_prototypes = defaultdict(list)
for doc_id, descriptor in doc_to_descriptor.items():
    writer = doc_to_writer.get(doc_id)
    if writer in test_writers:
        writer_prototypes[writer].append(descriptor)

# Average the descriptors for each writer
gallery_writers = list(writer_prototypes.keys())
gallery_vectors = np.array([np.mean(descs, axis=0) for descs in writer_prototypes.values()])
gallery_writer_map = {writer: i for i, writer in enumerate(gallery_writers)}

# --- Step 4: Perform Leave-One-Out Evaluation ---
print("Performing leave-one-out evaluation on the test set documents...")
average_precisions = []

# Iterate through all documents that belong to our test writers
for doc_id, query_desc in doc_to_descriptor.items():
    true_writer_id = doc_to_writer.get(doc_id)
    if true_writer_id not in test_writers:
        continue  # This is a training document, skip it.

    # Check if this writer has other documents in the test set to compare against
    num_relevant_docs = len(writer_prototypes.get(true_writer_id, [])) - 1
    if num_relevant_docs <= 0:
        continue  # Cannot evaluate if this is the only document for this writer in the test set

    # --- Create a temporary gallery for this query ---
    temp_gallery_vectors = gallery_vectors.copy()

    # Recalculate the prototype for the true writer, leaving out the query document
    other_descs = [d for d in writer_prototypes[true_writer_id] if not np.array_equal(d, query_desc)]
    true_writer_idx = gallery_writer_map[true_writer_id]
    temp_gallery_vectors[true_writer_idx] = np.mean(other_descs, axis=0)

    # Get similarities and ranked list of writers
    similarities = cosine_similarity(query_desc.reshape(1, -1), temp_gallery_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    ranked_writers = np.array(gallery_writers)[sorted_indices]

    # Calculate AP for this query
    hits = 0
    ap = 0.0
    for k, predicted_writer in enumerate(ranked_writers):
        if predicted_writer == true_writer_id:
            hits += 1
            ap += hits / (k + 1)

    average_precisions.append(ap / num_relevant_docs)

# --- Final Result ---
mean_ap = np.mean(average_precisions) * 100 if average_precisions else 0
print(f"\n✅ Final VLAD Result (Page-Independent Split):")
print(f"   Mean Average Precision (mAP): {mean_ap:.2f}%")