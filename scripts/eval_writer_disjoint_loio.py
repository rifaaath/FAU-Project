import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
from pathlib import Path


def get_doc_id_from_path(path_str):
    parts = Path(path_str).stem.split('_')
    return f"{parts[0]}_{parts[1]}" if len(parts) > 2 else Path(path_str).stem


def evaluate_disjoint_loio(args):
    # Load Data 
    print("Loading embeddings and manifest for the writer-disjoint test set...")
    embeddings_data = np.load(args.embeddings_path, allow_pickle=True)
    df_manifest = pd.read_csv(args.manifest_path)

    df = pd.DataFrame({'path': embeddings_data['paths'], 'embedding': list(embeddings_data['embeddings'])})
    df = pd.merge(df, df_manifest, on='path')

    # Generate doc_id on the fly
    df['doc_id'] = df['path'].apply(get_doc_id_from_path)

    # Create Page-Level Prototypes 
    print("Creating page-level style prototypes (mean pooling)...")
    page_prototypes = {
        doc_id: np.mean(np.stack(group['embedding'].values), axis=0)
        for doc_id, group in df.groupby('doc_id') if not group.empty
    }

    # Map doc_id to its ground truth writer
    ground_truth_map = dict(zip(df.doc_id, df.tm_id))
    doc_ids = sorted(list(page_prototypes.keys()))

    correct_predictions = 0
    total_queries = 0

    for query_doc_id in tqdm(doc_ids, desc="Evaluating LOIO on test pages"):
        query_vector = page_prototypes[query_doc_id]
        true_writer_id = ground_truth_map[query_doc_id]

        gallery_vectors, gallery_doc_ids = [], []
        for doc_id, vec in page_prototypes.items():
            if doc_id != query_doc_id:
                gallery_vectors.append(vec)
                gallery_doc_ids.append(doc_id)

        if not gallery_vectors: continue

        similarities = cosine_similarity(query_vector.reshape(1, -1), np.array(gallery_vectors))[0]
        predicted_idx = np.argmax(similarities)

        predicted_doc_id = gallery_doc_ids[predicted_idx]
        predicted_writer_id = ground_truth_map[predicted_doc_id]

        if predicted_writer_id == true_writer_id:
            correct_predictions += 1
        total_queries += 1

    accuracy = (correct_predictions / total_queries) * 100 if total_queries > 0 else 0.0
    print("\n" + "=" * 40)
    print("Writer-Independent Evaluation - Final Result")
    print("   Protocol: Leave-One-Image-Out Cross-Validation")
    print(f"   Top-1 Page-Level Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_queries})")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    args = parser.parse_args()
    evaluate_disjoint_loio(args)