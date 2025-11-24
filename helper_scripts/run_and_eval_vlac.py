# run_and_eval_vlac.py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import normalize
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def get_doc_id_from_path(path_str):
    """ A consistent function to get the document ID from a glyph path. """
    parts = Path(path_str).stem.split('_')
    # return f"{parts[0]}_{parts[1]}" if len(parts) > 2 else Path(path_str).stem
    return parts[0]

def run_vlac_pipeline(args):
    """
    A full, correct pipeline to build a VLAC codebook, create descriptors,
    and evaluate the page-level writer identification mAP.
    """
    # --- 1. Load All Necessary Data ---
    print("Loading embeddings and manifests...")
    try:
        full_embeddings_data = np.load(args.full_embeddings_path, allow_pickle=True)
        df_train_manifest = pd.read_csv(args.train_manifest_csv)
        df_test_manifest = pd.read_csv(args.test_manifest_csv)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Details: {e}")
        return

    # Create a fast lookup dictionary for embeddings
    path_to_embedding = {path: emb for path, emb in zip(full_embeddings_data['paths'], full_embeddings_data['embeddings'])}

    # --- 2. The Golden Rule: Strictly Separate Train and Test Data ---
    print("Separating training and test set embeddings...")
    train_embeddings = np.array([path_to_embedding[p] for p in df_train_manifest['path'] if p in path_to_embedding])
    df_train_manifest = df_train_manifest[df_train_manifest['path'].isin(path_to_embedding.keys())].copy() # Ensure manifest matches available embeddings

    test_embeddings = np.array([path_to_embedding[p] for p in df_test_manifest['path'] if p in path_to_embedding])
    df_test_manifest = df_test_manifest[df_test_manifest['path'].isin(path_to_embedding.keys())].copy()

    # Add document IDs for grouping
    df_test_manifest['doc_id'] = df_test_manifest['path'].apply(get_doc_id_from_path)
    
    print(f"Loaded {len(df_train_manifest)} glyphs for training codebook.")
    print(f"Loaded {len(df_test_manifest)} glyphs for testing.")

    # --- 3. Build the Codebook using ONLY Training Data ---
    print("Building character codebook from the TRAINING SET ONLY...")
    char_codebook = {}
    # Use 'base_char_name' for robust character classes
    for char_name, group in tqdm(df_train_manifest.groupby('base_char_name'), desc="Building codebook"):
        indices = group.index.values
        # Map manifest indices to the correct indices in the train_embeddings array
        embedding_indices = [df_train_manifest.index.get_loc(i) for i in indices]
        char_codebook[char_name] = np.mean(train_embeddings[embedding_indices], axis=0)
    
    valid_chars = sorted(list(char_codebook.keys()))
    print(f"Codebook created for {len(valid_chars)} character types.")
    
    # --- 4. Create VLAC Descriptors for Each Document in the TEST SET ---
    print("Creating VLAC descriptors for each document in the TEST SET...")
    page_descriptors = {}
    test_path_to_emb = {path: emb for path, emb in zip(df_test_manifest['path'], test_embeddings)}

    # Group the TEST manifest by document ID
    for doc_id, doc_group in tqdm(df_test_manifest.groupby('doc_id'), desc="Calculating VLACs"):
        doc_vlac_parts = []
        
        for char_name in valid_chars:
            # Get embeddings for this character ON THIS SPECIFIC PAGE
            page_char_paths = doc_group[doc_group['base_char_name'] == char_name]['path'].tolist()
            
            if page_char_paths:
                page_char_embs = np.array([test_path_to_emb[p] for p in page_char_paths])
                mean_vector = char_codebook[char_name]
                # This is the core logic: sum of residuals
                sum_of_residuals = np.sum(page_char_embs - mean_vector, axis=0)
            else:
                # If the character is not on the page, the residual sum is a zero vector
                sum_of_residuals = np.zeros_like(list(char_codebook.values())[0])
            
            doc_vlac_parts.append(sum_of_residuals)

        # Concatenate all parts into one long vector for the page
        full_vlac_vector = np.concatenate(doc_vlac_parts)
        
        # --- 5. Correctly Normalize the Final Vector ---
        # Power normalization (intra-normalization)
        normalized_vlac = np.sign(full_vlac_vector) * np.sqrt(np.abs(full_vlac_vector))
        # L2 normalization (global normalization)
        final_vlac = normalize(normalized_vlac.reshape(1, -1), norm='l2').flatten()
        
        page_descriptors[doc_id] = final_vlac

    # --- 6. Prepare for Evaluation ---
    print("Preparing data for mAP evaluation...")
    test_doc_ids = list(page_descriptors.keys())
    test_descriptors = np.array([page_descriptors[did] for did in test_doc_ids])
    
    # Create the ground truth mapping
    doc_to_writer_map = dict(zip(df_test_manifest.doc_id, df_test_manifest.tm_id))
    test_writer_ids = np.array([doc_to_writer_map[did] for did in test_doc_ids])

    # --- 7. Run mAP Evaluation ---
    print(f"Evaluating mAP on a test set of {len(test_descriptors)} pages.")
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
    
    # --- 8. FINAL RESULT ---
    print("\n" + "=" * 50)
    print("PAGE-LEVEL WRITER IDENTIFICATION RESULT (Simple-VLAC)")
    print(f"   Mean Average Precision (mAP): {mean_ap:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate the Simple-VLAC pipeline correctly.")
    parser.add_argument("--full_embeddings_path", type=str, required=True, help="Path to the .npz file with ALL embeddings.")
    parser.add_argument("--train_manifest_csv", type=str, required=True, help="Path to the TRAINING split manifest CSV.")
    parser.add_argument("--test_manifest_csv", type=str, required=True, help="Path to the TESTING split manifest CSV.")
    args = parser.parse_args()
    run_vlac_pipeline(args)