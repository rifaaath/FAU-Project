# eval_grk_loio.py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

def run_loio(args):
    # 1. Build Codebook from HomerComp (Training Data)
    print("Loading HomerComp Training Data...")
    homer_data = np.load(args.homer_embeddings, allow_pickle=True)
    homer_map = {p: e for p, e in zip(homer_data['paths'], homer_data['embeddings'])}
    df_homer = pd.read_csv(args.homer_manifest)
    
    print("Building Codebook...")
    codebook = {}
    for char, group in df_homer.groupby('base_char_name'):
        valid_paths = [p for p in group['path'] if p in homer_map]
        if not valid_paths: continue
        embs = np.array([homer_map[p] for p in valid_paths])
        codebook[char] = np.mean(embs, axis=0)
        
    # 2. Process GRK Pages
    print("Loading GRK Data...")
    grk_data = np.load(args.grk_embeddings, allow_pickle=True)
    grk_map = {p: e for p, e in zip(grk_data['paths'], grk_data['embeddings'])}
    df_grk = pd.read_csv(args.grk_manifest)
    
    page_descriptors = []
    page_writers = []
    page_ids = []
    
    # Group by Document (Page)
    for doc_id, group in df_grk.groupby('doc_id'):
        vlac_parts = []
        for char in sorted(codebook.keys()):
            char_group = group[group['base_char_name'] == char]
            valid_paths = [p for p in char_group['path'] if p in grk_map]
            
            if valid_paths:
                embs = np.array([grk_map[p] for p in valid_paths])
                # Sum of residuals
                residuals = np.sum(embs - codebook[char], axis=0)
                vlac_parts.append(residuals)
            else:
                vlac_parts.append(np.zeros_like(list(codebook.values())[0]))
                
        # Concatenate and Normalize
        full_vec = np.concatenate(vlac_parts)
        full_vec = np.sign(full_vec) * np.sqrt(np.abs(full_vec))
        full_vec = normalize(full_vec.reshape(1, -1)).flatten()
        
        page_descriptors.append(full_vec)
        page_writers.append(group['tm_id'].iloc[0])
        page_ids.append(doc_id)
        
    # 3. LOIO Evaluation
    print(f"Evaluating on {len(page_descriptors)} pages...")
    X = np.array(page_descriptors)
    y = np.array(page_writers)
    
    hits = 0
    for i in range(len(X)):
        query = X[i].reshape(1, -1)
        label = y[i]
        
        # Gallery is everyone else
        gallery_X = np.delete(X, i, axis=0)
        gallery_y = np.delete(y, i)
        
        sims = cosine_similarity(query, gallery_X)[0]
        top_idx = np.argmax(sims)
        pred_label = gallery_y[top_idx]
        
        if pred_label == label:
            hits += 1
            
    acc = hits / len(X)
    print(f"\nGRK Papyri 50 - Leave-One-Image-Out Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--homer_embeddings", required=True)
    parser.add_argument("--homer_manifest", required=True)
    parser.add_argument("--grk_embeddings", required=True)
    parser.add_argument("--grk_manifest", required=True)
    args = parser.parse_args()
    run_loio(args)