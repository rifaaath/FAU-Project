# evaluate_writer_disjoint_retrieval.py (FINAL, Corrected Variable Name)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# --- CONFIG ---
embedding_path = "embeddings.npz"  # This is the one for ALL glyphs
# --- END CONFIG ---

# --- Load Data ---
print("Loading data for evaluation...")
data = np.load(embedding_path, allow_pickle=True)
X = data["embeddings"]
paths = data["paths"]

# --- Get writer ID directly from the path ---
writer_ids = np.array([Path(p).parts[-2] for p in paths])

# --- The rest of the script now works perfectly with this correct ID ---
print(f"Found {len(np.unique(writer_ids))} unique writers in the dataset.")

unique_writers = np.unique(writer_ids)
np.random.seed(42)
np.random.shuffle(unique_writers)
test_writer_ids = unique_writers[int(0.8 * len(unique_writers)):]
print(f"Evaluating on {len(test_writer_ids)} unseen writers.")

test_mask = np.isin(writer_ids, test_writer_ids)
X_test = X[test_mask]
writer_ids_test = writer_ids[test_mask]

print("Building writer prototypes from glyph embeddings...")
writer_glyphs = defaultdict(list)
for emb, wid in zip(X_test, writer_ids_test):
    writer_glyphs[wid].append(emb)

gallery_writers = list(writer_glyphs.keys())
gallery_vectors = np.array([np.mean(embs, axis=0) for embs in writer_glyphs.values()])
gallery_writer_map = {writer: i for i, writer in enumerate(gallery_writers)}

print("Evaluating Top-1 retrieval accuracy...")
correct_predictions = 0
total_queries = 0

# --- ✅ CORRECTED VARIABLE NAME HERE ---
for query_wid, query_embs in tqdm(writer_glyphs.items(), desc="Evaluating writers"):
    if len(query_embs) < 2:
        continue

    for i, query_emb in enumerate(query_embs):
        temp_gallery_vectors = gallery_vectors.copy()
        other_embs = np.delete(query_embs, i, axis=0)
        writer_idx_in_gallery = gallery_writer_map[query_wid]
        temp_gallery_vectors[writer_idx_in_gallery] = np.mean(other_embs, axis=0)

        similarities = cosine_similarity(query_emb.reshape(1, -1), temp_gallery_vectors)[0]
        predicted_idx = np.argmax(similarities)
        predicted_writer = gallery_writers[predicted_idx]

        if predicted_writer == query_wid:
            correct_predictions += 1
        total_queries += 1

retrieval_accuracy = (correct_predictions / total_queries) * 100 if total_queries > 0 else 0
print(f"\n✅ Top-1 Writer-Disjoint Retrieval Accuracy (All Glyphs): {retrieval_accuracy:.2f}%")