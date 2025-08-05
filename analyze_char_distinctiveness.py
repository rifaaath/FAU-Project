# analyze_character_distinctiveness.py
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
embedding_path = "embeddings_page_independent.npz"
train_split_path = "final_splits/train_final.csv"
test_split_path = "final_splits/test_final.csv"
# The file created by compute_character_prototypes.py
prototype_path = "character_prototypes.npz"
output_csv_path = "writer_distinctiveness_analysis.csv"

# --- ❗️ CHOOSE THE CHARACTER YOU WANT TO ANALYZE ---
TARGET_CHAR_NAME = 'epsilon'
# --- End Config ---

# --- Step 1: Load Data and Prototypes ---
print("Loading data...")
prototypes_data = np.load(prototype_path, allow_pickle=True)
global_prototypes = prototypes_data['prototypes'].item()
global_mean_char = global_prototypes.get(TARGET_CHAR_NAME)

if global_mean_char is None:
    print(f"❌ Error: Could not find prototype for '{TARGET_CHAR_NAME}' in '{prototype_path}'")
    exit()

data = np.load(embedding_path, allow_pickle=True)
path_to_embedding = {path: emb for path, emb in zip(data["paths"], data["embeddings"])}
df_train = pd.read_csv(train_split_path)
df_test = pd.read_csv(test_split_path)
df_full = pd.concat([df_train, df_test])

# --- Step 2: Calculate each writer's average for the target character ---
writer_mean_chars = {}
all_writers = df_full['tm_id'].unique()

print(f"Calculating average '{TARGET_CHAR_NAME}' for each writer...")
for writer in tqdm(all_writers, desc="Processing writers"):
    # Filter to get all glyphs for this writer and this character
    writer_char_df = df_full[(df_full['tm_id'] == writer) & (df_full['base_char_name'] == TARGET_CHAR_NAME)]

    writer_embs = [path_to_embedding[p] for p in writer_char_df['path'] if p in path_to_embedding]

    if len(writer_embs) >= 5:  # Only consider writers with a minimum number of samples
        writer_mean_chars[writer] = np.mean(writer_embs, axis=0)

# --- Step 3: Calculate the distance of each writer's average to the global average ---
results = []
print(f"\nCalculating distinctiveness from global mean '{TARGET_CHAR_NAME}'...")
for writer, writer_mean in tqdm(writer_mean_chars.items(), desc="Calculating distances"):
    distance = 1 - cosine_similarity(writer_mean.reshape(1, -1), global_mean_char.reshape(1, -1))[0, 0]
    results.append({
        "Writer_ID": writer,
        "Distinctiveness_Score": distance,
        "Num_Samples": len(df_full[(df_full['tm_id'] == writer) & (df_full['base_char_name'] == TARGET_CHAR_NAME)])
    })

# --- Step 4: Save to CSV for analysis ---
if results:
    df_results = pd.DataFrame(results)
    # Sort by the most distinctive (highest distance) writers first
    df_results = df_results.sort_values(by="Distinctiveness_Score", ascending=False)
    df_results.to_csv(output_csv_path, index=False)
    print(f"\n✅ Analysis complete. Results saved to '{output_csv_path}'")
    print(f"\n--- Top 5 Most Distinctive Writers for '{TARGET_CHAR_NAME.capitalize()}' ---")
    print(df_results.head(5).to_string())
else:
    print("\nCould not find enough data to perform analysis.")