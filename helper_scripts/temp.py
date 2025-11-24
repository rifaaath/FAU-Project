# debug_path_mismatch.py
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Debug path mismatch issues.")
parser.add_argument("--manifest_csv", type=str, required=True)
parser.add_argument("--embeddings_npz", type=str, required=True)
args = parser.parse_args()

print("--- Loading Manifest CSV ---")
df_manifest = pd.read_csv(args.manifest_csv)
manifest_paths = df_manifest['path'].tolist()
print(f"Found {len(manifest_paths)} paths in {args.manifest_csv}")

print("\n--- Loading Embeddings NPZ ---")
embeddings_data = np.load(args.embeddings_npz, allow_pickle=True)
embedding_paths = embeddings_data['paths'].tolist()
print(f"Found {len(embedding_paths)} paths in {args.embeddings_npz}")

print("\n--- COMPARING THE FIRST 5 PATHS ---")
print("These paths MUST be identical strings to match.\n")

for i in range(5):
    if i < len(manifest_paths) and i < len(embedding_paths):
        print(f"Entry #{i+1}:")
        print(f"  Manifest Path:  '{manifest_paths[i]}'")
        print(f"  Embedding Path: '{embedding_paths[i]}'")
        if manifest_paths[i] == embedding_paths[i]:
            print("  Result: MATCH ✅")
        else:
            print("  Result: MISMATCH ❌")
        print("-" * 20)