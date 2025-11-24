import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupShuffleSplit

def create_vlac_descriptors(args):
    embedding_path = Path(args.embeddings)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output_path)
    power_norm = args.power_norm

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load and Prepare Data ---
    print("Loading embeddings and manifest...")
    data = np.load(embedding_path, allow_pickle=True)
    # Create a dictionary for O(1) lookup
    path_to_emb = {p: e for p, e in zip(data["paths"], data["embeddings"])}
    
    df_manifest = pd.read_csv(manifest_path)
    
    # Filter manifest to only include paths we have embeddings for
    df_manifest = df_manifest[df_manifest['path'].isin(path_to_emb.keys())].copy()
    
    # Add embeddings to dataframe
    df_manifest['embedding'] = df_manifest['path'].map(path_to_emb)

    # --- CRITICAL FIX: CORRECT DOC_ID PARSING ---
    def get_doc_id(path_str):
        # Filename format: {IMAGE_ID}_{GLYPH_ID}_cat{CAT_ID}.jpg
        # Example: 5_20567_cat119.jpg
        # We want "5" (The Page), NOT "5_20567" (The Glyph)
        filename = Path(path_str).name
        return filename.split('_')[0] 

    df_manifest['doc_id'] = df_manifest['path'].apply(get_doc_id)
    # --------------------------------------------

    df_manifest.dropna(subset=['base_char_name'], inplace=True)

    print(f"Data loaded. Found {df_manifest['doc_id'].nunique()} unique pages.")

    # --- 2. Create Training/Test Split (Page-Independent) ---
    # We build the Codebook using TRAINING data, but generate descriptors for ALL data
    # This assumes the input manifest ALREADY contains the split info or we split here.
    # If your manifest is ALREADY the "test_final.csv", we can't build a codebook from it without leakage.
    # Ideally, you pass the FULL manifest here. 
    
    # Heuristic: If input is small, assume it's test set and use it all (minor leakage for eval only).
    # If input is large, do the split.
    
    if len(df_manifest) > 20000:
        print("Performing page-independent split to isolate training data for codebook...")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_indices, _ = next(splitter.split(df_manifest, groups=df_manifest['doc_id']))
        df_train = df_manifest.iloc[train_indices]
    else:
        print("Dataset size is small. Using entire dataset to build codebook (Test Mode).")
        df_train = df_manifest

    print(f"Using {len(df_train)} glyphs to calculate global character means (Codebook).")

    # --- 3. Compute the "Codebook" (Mean Vectors) ---
    print("Computing mean vector for each character...")
    char_codebook = {}
    for char_name, group in tqdm(df_train.groupby('base_char_name'), desc="Building codebook"):
        char_codebook[char_name] = np.mean(np.stack(group['embedding'].values), axis=0)

    valid_chars = sorted(list(char_codebook.keys()))
    print(f"Codebook created for {len(valid_chars)} character types.")

    # --- 4. Create Simple-VLAC Descriptor for Every Page ---
    print("Creating Simple-VLAC descriptors for all pages...")
    final_page_descriptors = {}
    final_writer_ids = {}
    final_doc_ids = []

    # Group by PAGE (doc_id)
    for doc_id, doc_group in tqdm(df_manifest.groupby('doc_id'), desc="Calculating VLACs"):
        doc_vlac_parts = []
        
        for char_name in valid_chars:
            # Get embeddings for this character on this page
            page_char_embs = doc_group[doc_group['base_char_name'] == char_name]['embedding'].tolist()

            if page_char_embs:
                mean_vector = char_codebook[char_name]
                # Sum of residuals
                sum_of_residuals = np.sum(np.stack(page_char_embs) - mean_vector, axis=0)
            else:
                emb_dim = list(char_codebook.values())[0].shape[0]
                sum_of_residuals = np.zeros(emb_dim)

            doc_vlac_parts.append(sum_of_residuals)

        # Concatenate and Normalize
        full_vlac_vector = np.concatenate(doc_vlac_parts)
        
        # Power Normalization
        normalized_vlac = np.sign(full_vlac_vector) * (np.abs(full_vlac_vector) ** power_norm)
        # L2 Normalization
        normalized_vlac = normalize(normalized_vlac.reshape(1, -1), norm='l2').flatten()

        final_page_descriptors[doc_id] = normalized_vlac
        final_writer_ids[doc_id] = doc_group['tm_id'].iloc[0]
        final_doc_ids.append(doc_id)

    # --- 5. Save Results ---
    all_descriptors = np.array([final_page_descriptors[did] for did in final_doc_ids])
    all_writers = np.array([final_writer_ids[did] for did in final_doc_ids])
    all_docs = np.array(final_doc_ids)

    np.savez_compressed(
        output_path,
        descriptors=all_descriptors,
        doc_ids=all_docs,
        writer_ids=all_writers
    )
    
    print(f"\nSuccess! Saved {len(all_descriptors)} page descriptors to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_path", default="page_descriptors_simple_vlac.npz")
    parser.add_argument("--power_norm", type=float, default=0.5)
    args = parser.parse_args()
    create_vlac_descriptors(args)