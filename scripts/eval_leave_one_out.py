import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
import argparse


def evaluate_loio_cv(args):
    # Load Data 
    print("Loading embeddings and manifest...")
    embeddings_data = np.load(args.embeddings_path, allow_pickle=True)
    df_manifest = pd.read_csv(args.manifest_path)

    # Create a DataFrame with all data
    df = pd.DataFrame({
        'path': embeddings_data['paths'],
        'embedding': list(embeddings_data['embeddings'])
    })
    df = pd.merge(df, df_manifest, on='path')

    # Create Page-Level Prototypes (Average Hand for Each Image) 
    print("Creating page-level style prototypes (mean pooling)...")
    page_prototypes = {}
    for image_filename, group in df.groupby('image_filename'):
        if not group.empty:
            page_prototypes[image_filename] = np.mean(np.stack(group['embedding'].values), axis=0)

    # Run Leave-One-Image-Out Cross-Validation 
    print("Running Leave-One-Image-Out evaluation...")
    image_filenames = sorted(list(page_prototypes.keys()))
    ground_truth_map = dict(zip(df.image_filename, df.tm_id))

    correct_predictions = 0
    total_queries = 0

    for query_filename in tqdm(image_filenames, desc="Evaluating"):
        query_vector = page_prototypes[query_filename]
        true_writer_id = ground_truth_map[query_filename]

        gallery_vectors = []
        gallery_filenames = []

        # Build the gallery from all *other* images
        for fname, vec in page_prototypes.items():
            if fname != query_filename:
                gallery_vectors.append(vec)
                gallery_filenames.append(fname)

        if not gallery_vectors:
            continue  # Should not happen in this dataset

        gallery_vectors = np.array(gallery_vectors)

        # Calculate similarity and find the top match
        similarities = cosine_similarity(query_vector.reshape(1, -1), gallery_vectors)[0]
        predicted_idx = np.argmax(similarities)

        predicted_filename = gallery_filenames[predicted_idx]
        predicted_writer_id = ground_truth_map[predicted_filename]

        if predicted_writer_id == true_writer_id:
            correct_predictions += 1
        total_queries += 1

    # Final Result 
    accuracy = (correct_predictions / total_queries) * 100 if total_queries > 0 else 0.0
    print("\n" + "=" * 40)
    print("GRK Papyri 50 - Final Result")
    print("   Protocol: Leave-One-Image-Out Cross-Validation")
    print(f"   Top-1 Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_queries})")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    args = parser.parse_args()
    evaluate_loio_cv(args)