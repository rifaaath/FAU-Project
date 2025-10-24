import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pathlib import Path
import argparse

def evaluate(args):
    test_embedding_path = Path(args.test_embedding_path)
    print(f"Loading TEST data from: {test_embedding_path}")
    try:
        test_data = np.load(test_embedding_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: The file '{test_embedding_path}' was not found.")
        exit()

    gallery_embs = test_data["embeddings"]
    gallery_paths = test_data["paths"]
    gallery_writers = np.array([Path(p).parts[-2] for p in gallery_paths])
    print(f"Building gallery from {len(gallery_embs)} glyphs in the test set.")

    print("Performing leave-one-out evaluation on the test set...")
    average_precisions = []
    for i in tqdm(range(len(gallery_embs)), desc="Evaluating Queries"):
        query_emb = gallery_embs[i]
        true_writer_id = gallery_writers[i]
        temp_gallery_embs = np.delete(gallery_embs, i, axis=0)
        temp_gallery_writers = np.delete(gallery_writers, i)
        num_relevant_docs = np.sum(temp_gallery_writers == true_writer_id)
        if num_relevant_docs == 0: continue
        similarities = cosine_similarity(query_emb.reshape(1, -1), temp_gallery_embs)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        ranked_writers = temp_gallery_writers[sorted_indices]
        hits = 0
        ap = 0.0
        for k, predicted_writer in enumerate(ranked_writers):
            if predicted_writer == true_writer_id:
                hits += 1
                ap += hits / (k + 1)
        average_precisions.append(ap / num_relevant_docs)

    mean_ap = np.mean(average_precisions) * 100 if average_precisions else 0
    print(f"\n{'='*40}\nFINAL RESULT (Page-Independent Split):\n{'='*40}")
    print(f"   Mean Average Precision (mAP): {mean_ap:.2f}%")
    print(f"{'='*40}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate writer retrieval mAP on a set of test embeddings.")
    parser.add_argument("--test_embedding_path", type=str, required=True)
    args = parser.parse_args()
    evaluate(args)