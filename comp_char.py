# compute_character_prototypes.py
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# --- Config ---
# Use your best, page-independent embeddings
embedding_path = "embeddings_page_independent.npz"
# We need the final, clean splits to get the character names and IDs
train_split_path = "final_splits/train_final.csv"
test_split_path = "final_splits/test_final.csv"

output_path = "character_prototypes.npz"
# --- End Config ---

# --- Step 1: Load embeddings and create a path-to-embedding lookup dictionary ---
print(f"Loading data from {embedding_path}...")
data = np.load(embedding_path, allow_pickle=True)
embeddings = data["embeddings"]
paths = data["paths"]
path_to_embedding = {path: emb for path, emb in zip(paths, embeddings)}

# --- Step 2: Load the split files to get the full list of glyphs and their metadata ---
print("Loading manifest data from splits...")
df_train = pd.read_csv(train_split_path)
df_test = pd.read_csv(test_split_path)
df_full = pd.concat([df_train, df_test])

# Ensure the required column exists
if 'base_char_name' not in df_full.columns:
    print("❌ Error: The manifest is missing the 'base_char_name' column.")
    print("Please run the 'prepare_final_dataset.py' script first to generate the correct splits.")
    exit()

print(f"Loaded metadata for {len(df_full)} glyphs.")

# --- Step 3: Group embeddings by their base character name ---
embs_by_char_name = defaultdict(list)
print("Grouping embeddings by character...")
# Iterate through the DataFrame for metadata
for _, row in tqdm(df_full.iterrows(), total=len(df_full)):
    # Check if we have an embedding for this path
    if row['path'] in path_to_embedding:
        char_name = row['base_char_name']
        embs_by_char_name[char_name].append(path_to_embedding[row['path']])

# --- Step 4: Calculate the mean embedding (prototype) for each character class ---
char_prototypes = {}
print("\nCalculating mean prototype for each character...")
for char_name, embs in tqdm(embs_by_char_name.items()):
    if embs:
        # The prototype is the L2-normalized mean of all embeddings for that char
        mean_vec = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            char_prototypes[char_name] = mean_vec / norm
        else:
            char_prototypes[char_name] = mean_vec

# --- Step 5: Save the prototypes ---
# We save as a single dictionary-like object for easy lookup by character name
# np.savez_compressed allows saving non-array objects like dictionaries
np.savez_compressed(
    output_path,
    prototypes=char_prototypes
)

print(f"\n✅ Success! Saved {len(char_prototypes)} character prototypes to '{output_path}'")
print("\n--- Example Prototypes (Top 5) ---")
for i, (name, proto) in enumerate(list(char_prototypes.items())[:5]):
    print(f"  - Character: '{name}', Prototype Shape: {proto.shape}")