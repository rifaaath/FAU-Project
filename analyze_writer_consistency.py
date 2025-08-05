# analyze_writer_consistency.py
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
embedding_path = "embeddings_page_independent.npz"
# Use the final, clean, and normalized splits
train_split_path = "final_splits/train_final.csv"
test_split_path = "final_splits/test_final.csv"

output_csv_path = "writer_consistency_analysis.csv"

# --- ❗️ CHOOSE THE TWO CHARACTERS YOU WANT TO COMPARE ---
# Use the base character names from your dataset (e.g., 'epsilon', 'alpha')
CHAR_A_NAME = 'epsilon'
CHAR_B_NAME = 'alpha'
# --- End Config ---

# --- Step 1: Load Data ---
print("Loading data...")
data = np.load(embedding_path, allow_pickle=True)
path_to_embedding = {path: emb for path, emb in zip(data["paths"], data["embeddings"])}

df_train = pd.read_csv(train_split_path)
df_test = pd.read_csv(test_split_path)
df_full = pd.concat([df_train, df_test])

# --- Step 2: Group embeddings by (writer, character) ---
embs_by_writer_char = defaultdict(list)
for _, row in tqdm(df_full.iterrows(), total=len(df_full), desc="Grouping data"):
    if row['path'] in path_to_embedding:
        key = (row['tm_id'], row['base_char_name'])
        embs_by_writer_char[key].append(path_to_embedding[row['path']])

# --- Step 3: Calculate Consistency and Distance for Each Writer ---
results = []
all_writers = df_full['tm_id'].unique()
print(f"\nAnalyzing consistency for {len(all_writers)} writers...")

for writer in tqdm(all_writers, desc="Analyzing writers"):
    # Get all glyphs for Char A (e.g., Epsilon) for this writer
    writer_char_a_embs = embs_by_writer_char.get((writer, CHAR_A_NAME), [])
    # Get all glyphs for Char B (e.g., Alpha) for this writer
    writer_char_b_embs = embs_by_writer_char.get((writer, CHAR_B_NAME), [])

    # We need at least a few samples of each to do a meaningful comparison
    if len(writer_char_a_embs) < 5 or len(writer_char_b_embs) < 5:
        continue

    # --- Metric 1: Intra-Class Cohesion ---
    # How similar are this writer's epsilons to their own average epsilon?
    mean_char_a = np.mean(writer_char_a_embs, axis=0).reshape(1, -1)
    cohesion_char_a = np.mean(cosine_similarity(writer_char_a_embs, mean_char_a))

    # --- Metric 2: Inter-Class Distance ---
    # How different is this writer's average epsilon from their average alpha?
    mean_char_b = np.mean(writer_char_b_embs, axis=0).reshape(1, -1)
    # Cosine distance = 1 - cosine similarity
    distance_a_b = 1 - cosine_similarity(mean_char_a, mean_char_b)[0, 0]

    results.append({
        "Writer_ID": writer,
        f"{CHAR_A_NAME.capitalize()}_Cohesion": cohesion_char_a,
        f"{CHAR_A_NAME.capitalize()}_{CHAR_B_NAME.capitalize()}_Distance": distance_a_b,
        f"Num_{CHAR_A_NAME.capitalize()}": len(writer_char_a_embs),
        f"Num_{CHAR_B_NAME.capitalize()}": len(writer_char_b_embs)
    })

# --- Step 4: Save to CSV for analysis ---
if results:
    df_results = pd.DataFrame(results)
    # Sort by the most distinct writers first
    df_results = df_results.sort_values(by=f"{CHAR_A_NAME.capitalize()}_{CHAR_B_NAME.capitalize()}_Distance", ascending=False)
    df_results.to_csv(output_csv_path, index=False)
    print(f"\n✅ Analysis complete. Results saved to '{output_csv_path}'")
    print("\n--- Top 5 Most Distinct Writers (Epsilon vs Alpha) ---")
    print(df_results.head(5).to_string())
else:
    print("\nCould not find enough data for any writer to perform the analysis.")