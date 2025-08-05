# evaluate_retrieval_with_map.py (FINAL - Uses separate test embeddings)
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# --- Config ---
# --- ✅ WE NOW LOAD THE DEDICATED TEST EMBEDDINGS ---
test_embedding_path = "split_embeddings.npz"
# --- End Config ---

# --- Load the Test Data ---
print(f"Loading TEST data from: {test_embedding_path}")
try:
    test_data = np.load(test_embedding_path, allow_pickle=True)
except FileNotFoundError:
    print(f"❌ Error: The file '{test_embedding_path}' was not found.")
    print("Please run `extract_embeddings.py` on your 'test_split.csv' to generate it.")
    exit()

# The gallery is simply all the embeddings and paths from the test file.
gallery_embs = test_data["embeddings"]
gallery_paths = test_data["paths"]

# --- ✅ Get writer IDs directly from the paths in the test file ---
gallery_writers = np.array([Path(p).parts[-2] for p in gallery_paths])

print(f"Building gallery from {len(gallery_embs)} glyphs in the test set.")

# --- Perform Leave-One-Out Cross-Validation on the Test Set ---
print("Performing leave-one-out evaluation on the test set...")
average_precisions = []

for i in tqdm(range(len(gallery_embs)), desc="Evaluating Queries"):
    query_emb = gallery_embs[i]
    true_writer_id = gallery_writers[i]

    # Create a temporary gallery by removing the i-th element
    temp_gallery_embs = np.delete(gallery_embs, i, axis=0)
    temp_gallery_writers = np.delete(gallery_writers, i)

    # Check if there are any other glyphs from the same writer left to be found
    num_relevant_docs = np.sum(temp_gallery_writers == true_writer_id)
    if num_relevant_docs == 0:
        continue  # Cannot calculate AP if there are no other positive examples

    # Calculate similarity scores
    similarities = cosine_similarity(query_emb.reshape(1, -1), temp_gallery_embs)[0]

    # Get the ranked list of writers
    sorted_indices = np.argsort(similarities)[::-1]
    ranked_writers = temp_gallery_writers[sorted_indices]

    # --- Calculate Average Precision (AP) for this single query ---
    hits = 0
    ap = 0.0
    for k, predicted_writer in enumerate(ranked_writers):
        if predicted_writer == true_writer_id:
            hits += 1
            ap += hits / (k + 1)

    average_precisions.append(ap / num_relevant_docs)

# --- Final Result ---
mean_ap = np.mean(average_precisions) * 100 if average_precisions else 0
print(f"\n✅ Final Result (Page-Independent Split):")
print(f"   Mean Average Precision (mAP): {mean_ap:.2f}%")