import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

def run_cross_dataset_vlac(args):
    print(f"--- CROSS-DATASET VLAC EVALUATION ---")
    print(f"Source (Codebook): HomerComp")
    print(f"Target (Evaluation): GRK Papyri 50")

    # ==========================================
    # 1. Build Codebook from HomerComp (Train)
    # ==========================================
    print("\n[1/4] Loading HomerComp Training Data...")
    homer_data = np.load(args.homer_embeddings, allow_pickle=True)
    # Fast lookup
    homer_emb_map = {p: e for p, e in zip(homer_data['paths'], homer_data['embeddings'])}
    
    df_homer_train = pd.read_csv(args.homer_manifest)
    
    print("      Building Character Codebook (Means)...")
    codebook = {}
    
    # Group by character class to calculate means
    # We only use the TRAINING set of HomerComp
    grouped = df_homer_train.groupby('base_char_name')
    
    for char_name, group in grouped:
        # Collect valid embeddings
        embs = []
        for path in group['path']:
            if path in homer_emb_map:
                embs.append(homer_emb_map[path])
        
        if len(embs) > 0:
            codebook[char_name] = np.mean(np.array(embs), axis=0)

    valid_chars = sorted(list(codebook.keys()))
    print(f"      Codebook built for {len(valid_chars)} character types.")

    # ==========================================
    # 2. Load GRK Data
    # ==========================================
    print("\n[2/4] Loading GRK Papyri Data...")
    grk_data = np.load(args.grk_embeddings, allow_pickle=True)
    grk_emb_map = {p: e for p, e in zip(grk_data['paths'], grk_data['embeddings'])}
    
    df_grk = pd.read_csv(args.grk_manifest)
    
    # Filter GRK manifest to only include valid embeddings
    df_grk = df_grk[df_grk['path'].isin(grk_emb_map.keys())].copy()
    
    print(f"      Found {len(df_grk)} valid GRK glyphs.")
    print(f"      Found {df_grk['doc_id'].nunique()} unique pages (documents).")

    # ==========================================
    # 3. Generate VLAC Descriptors for GRK Pages
    # ==========================================
    print("\n[3/4] Generating VLAC Descriptors using HomerComp Codebook...")
    
    page_descriptors = []
    page_writers = []
    page_ids = []

    # Group GRK data by PAGE (doc_id)
    for doc_id, page_group in tqdm(df_grk.groupby('doc_id')):
        vlac_parts = []
        
        # Iterate through the HOMERCOMP character list
        for char_name in valid_chars:
            # Find GRK glyphs that match this character
            page_char_paths = page_group[page_group['base_char_name'] == char_name]['path'].tolist()
            
            if page_char_paths:
                # Get GRK embeddings
                embs = np.array([grk_emb_map[p] for p in page_char_paths])
                
                # --- THE CORE VLAC MATH ---
                # Sum of Residuals: (GRK Glyph) - (Homer Mean)
                # This measures how the GRK writer deviates from the Homer average
                residuals = np.sum(embs - codebook[char_name], axis=0)
                vlac_parts.append(residuals)
            else:
                # Character missing on this page
                emb_dim = list(codebook.values())[0].shape[0]
                vlac_parts.append(np.zeros(emb_dim))
        
        # Concatenate
        full_vec = np.concatenate(vlac_parts)
        
        # Normalize (Power + L2)
        full_vec = np.sign(full_vec) * np.sqrt(np.abs(full_vec))
        full_vec = normalize(full_vec.reshape(1, -1)).flatten()
        
        page_descriptors.append(full_vec)
        page_writers.append(page_group['tm_id'].iloc[0])
        page_ids.append(doc_id)

    # ==========================================
    # 4. Leave-One-Image-Out Evaluation
    # ==========================================
    print("\n[4/4] Running Leave-One-Image-Out (LOIO) Evaluation...")
    
    X = np.array(page_descriptors)
    y = np.array(page_writers)
    n_pages = len(X)
    
    hits = 0
    
    for i in range(n_pages):
        query_vec = X[i].reshape(1, -1)
        true_writer = y[i]
        
        # Gallery is everyone else
        gallery_vecs = np.delete(X, i, axis=0)
        gallery_writers = np.delete(y, i)
        
        # Skip if this writer has no other pages (cannot be matched)
        if np.sum(gallery_writers == true_writer) == 0:
            continue
            
        # Nearest Neighbor
        sims = cosine_similarity(query_vec, gallery_vecs)[0]
        best_idx = np.argmax(sims)
        pred_writer = gallery_writers[best_idx]
        
        if pred_writer == true_writer:
            hits += 1
            
    accuracy = hits / n_pages
    print("\n" + "="*50)
    print(f"FINAL RESULT: GRK Papyri 50 (Zero-Shot Transfer)")
    print(f"Method: Simple-VLAC (HomerComp Means)")
    print(f"Top-1 Accuracy: {accuracy*100:.2f}% ({hits}/{n_pages})")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--homer_embeddings", required=True)
    parser.add_argument("--homer_manifest", required=True)
    parser.add_argument("--grk_embeddings", required=True)
    parser.add_argument("--grk_manifest", required=True)
    args = parser.parse_args()
    run_cross_dataset_vlac(args)