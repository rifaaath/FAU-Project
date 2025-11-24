# benchmark_2_eval_vlad.py
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
import argparse
from pathlib import Path

def compute_vlad(features, labels, sids, class_centers):
    """
    Computes a VLAD descriptor for each unique sample ID (document).
    """
    # Group features and labels by their sample/document ID
    sample_feats = defaultdict(list)
    for f, l, sid in zip(features, labels, sids):
        sample_feats[sid].append((f, l))

    encs, order = [], []
    for sid, feat_label_list in tqdm(sample_feats.items(), desc="Computing VLAD descriptors"):
        by_class = defaultdict(list)
        for f, l in feat_label_list:
            by_class[l].append(f)

        parts = []
        for c in sorted(class_centers):
            mu = class_centers[c]
            if c in by_class:
                # Sum of residuals
                v = np.sum(np.array(by_class[c]) - mu, axis=0)
            else:
                v = np.zeros_like(mu)
            parts.append(v)

        # Concatenate, power-normalize, and L2-normalize
        vlad = np.concatenate(parts)
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        norm = np.linalg.norm(vlad)
        if norm > 1e-10:
            vlad /= norm

        encs.append(vlad)
        order.append(sid)

    return np.array(encs), np.array(order)


def evaluate_hog_vlad(args):
    # --- 1. Load Data ---
    print("Loading pre-computed HOG features...")
    with open(args.train_features_pkl, "rb") as f:
        train_data = pickle.load(f)
    X_train, y_train = train_data["features"], train_data["labels"]

    with open(args.test_features_pkl, "rb") as f:
        test_data = pickle.load(f)
    X_test, y_test, sid_test = test_data["features"], test_data["labels"], test_data["sample_ids"]

    # --- 2. PCA ---
    print("Training PCA on training data and transforming features...")
    pca = PCA(n_components=0.95, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA reduced feature dimension to: {X_train_pca.shape[1]}")

    # --- 3. Build Codebook from Training Data ---
    print("Building character codebook from training set...")
    class_centers = {}
    for c in tqdm(np.unique(y_train), desc="Calculating class centers"):
        class_centers[c] = X_train_pca[y_train == c].mean(axis=0)

    # --- 4. Compute VLAD for Test Set ---
    print("Computing VLAD descriptors for test set documents...")
    test_vlad_vectors, test_doc_ids = compute_vlad(X_test_pca, y_test, sid_test, class_centers)

    # --- 5. Prepare for Evaluation ---
    print("Mapping documents to ground truth writer IDs...")
    df_test_manifest = pd.read_csv(args.test_manifest_csv)
    df_test_manifest['doc_id'] = df_test_manifest['path'].apply(
        lambda p: f"{Path(p).stem.split('_')[0]}_{Path(p).stem.split('_')[1]}"
    )
    doc_to_writer_map = dict(zip(df_test_manifest.doc_id, df_test_manifest.tm_id))
    ground_truth_writers = np.array([doc_to_writer_map.get(doc_id, 'N/A') for doc_id in test_doc_ids])

    # --- 6. Run Page-Level Evaluation ---
    print("Running Leave-One-Out evaluation on test set pages...")
    correct_predictions = 0
    total_queries = len(test_vlad_vectors)

    for i in tqdm(range(total_queries), desc="Evaluating"):
        query_vector = test_vlad_vectors[i].reshape(1, -1)
        true_writer_id = ground_truth_writers[i]

        # Build gallery by excluding the query
        gallery_vectors = np.delete(test_vlad_vectors, i, axis=0)
        gallery_writers = np.delete(ground_truth_writers, i)

        # Calculate similarity and find the top match
        similarities = cosine_similarity(query_vector, gallery_vectors)[0]
        predicted_idx = np.argmax(similarities)
        predicted_writer_id = gallery_writers[predicted_idx]

        if predicted_writer_id == true_writer_id:
            correct_predictions += 1

    # --- 7. Final Result ---
    accuracy = (correct_predictions / total_queries) * 100 if total_queries > 0 else 0.0
    print("\n" + "=" * 50)
    print("HOG + VLAD BENCHMARK RESULT")
    print(f"   Protocol: Writer-Disjoint, Page-Level Identification")
    print(f"   Top-1 Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_queries})")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate writer ID using HOG+VLAD.")
    parser.add_argument("--train_features_pkl", type=str, required=True, help="Path to the HOG features of the training set.")
    parser.add_argument("--test_features_pkl", type=str, required=True, help="Path to the HOG features of the test set.")
    parser.add_argument("--test_manifest_csv", type=str, required=True, help="Path to the test manifest CSV for getting writer IDs.")
    args = parser.parse_args()
    evaluate_hog_vlad(args)
