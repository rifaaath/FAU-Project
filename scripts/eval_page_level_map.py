import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

def evaluate_page_map(args):
    # Load the VLAC descriptors
    data = np.load(args.descriptors_path, allow_pickle=True)
    descriptors = data['descriptors']
    doc_ids = data['doc_ids']
    writer_ids = data['writer_ids']

    # We need to evaluate only on the TEST set to be fair.
    # We recreate the same split used in generation.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_indices = next(splitter.split(descriptors, groups=doc_ids))

    test_descriptors = descriptors[test_indices]
    test_writers = writer_ids[test_indices]
    
    print(f"Evaluating mAP on {len(test_descriptors)} test pages...")

    average_precisions = []
    
    # Leave-One-Out Evaluation on Page Level
    for i in tqdm(range(len(test_descriptors))):
        query_vec = test_descriptors[i].reshape(1, -1)
        true_writer = test_writers[i]

        # Gallery is all other test pages
        gallery_vecs = np.delete(test_descriptors, i, axis=0)
        gallery_writers = np.delete(test_writers, i)

        # Skip if this writer has no other pages in test set (can happen with small data)
        if np.sum(gallery_writers == true_writer) == 0:
            continue

        # Calculate Similarity
        similarities = cosine_similarity(query_vec, gallery_vecs)[0]
        
        # Rank results
        indices = np.argsort(similarities)[::-1]
        ranked_writers = gallery_writers[indices]

        # Calculate AP
        hits = 0
        sum_precisions = 0
        num_relevant = 0
        
        for k, w in enumerate(ranked_writers):
            if w == true_writer:
                hits += 1
                num_relevant += 1
                sum_precisions += hits / (k + 1)
        
        if num_relevant > 0:
            average_precisions.append(sum_precisions / num_relevant)

    mean_ap = np.mean(average_precisions) * 100 if average_precisions else 0
    print(f"\nPage-Level mAP (Simple-VLAC): {mean_ap:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors_path", required=True)
    args = parser.parse_args()
    evaluate_page_map(args)