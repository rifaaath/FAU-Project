import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
from sklearn.model_selection import GroupShuffleSplit


def evaluate_page_map(args):
    # Load Data 
    data = np.load(args.descriptors_path, allow_pickle=True)
    descriptors = data['descriptors']
    doc_ids = data['doc_ids']
    writer_ids = data['writer_ids']

    # Perform the same Page-Independent Split 
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_indices = next(splitter.split(descriptors, groups=doc_ids))

    test_descriptors = descriptors[test_indices]
    test_writer_ids = writer_ids[test_indices]

    print(f"Evaluating mAP on a test set of {len(test_descriptors)} pages.")

    # Run mAP Evaluation 
    average_precisions = []
    for i in tqdm(range(len(test_descriptors)), desc="Evaluating Queries"):
        query_desc = test_descriptors[i]
        true_writer_id = test_writer_ids[i]

        gallery_descs = np.delete(test_descriptors, i, axis=0)
        gallery_writers = np.delete(test_writer_ids, i)

        num_relevant_docs = np.sum(gallery_writers == true_writer_id)
        if num_relevant_docs == 0: continue

        similarities = cosine_similarity(query_desc.reshape(1, -1), gallery_descs)[0]
        ranked_writers = gallery_writers[np.argsort(similarities)[::-1]]

        hits, ap = 0, 0.0
        for k, pred_writer in enumerate(ranked_writers):
            if pred_writer == true_writer_id:
                hits += 1
                ap += hits / (k + 1)

        average_precisions.append(ap / num_relevant_docs)

    mean_ap = np.mean(average_precisions) * 100 if average_precisions else 0.0
    print("\n" + "=" * 40)
    print("Page-Level Identification Result")
    print(f"   Aggregation Method: Simple-VLAC")
    print(f"   Mean Average Precision (mAP): {mean_ap:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate page-level descriptors using mAP.")
    parser.add_argument("--descriptors_path", required=True)
    args = parser.parse_args()
    evaluate_page_map(args)