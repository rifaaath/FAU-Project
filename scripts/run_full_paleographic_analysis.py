import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import json

embedding_path = "embeddings_page_independent.npz"
train_split_path = "final_splits/train_final.csv"
test_split_path = "final_splits/test_final.csv"

TARGET_CHAR_A = 'epsilon'
TARGET_CHAR_B = 'alpha'
MIN_SAMPLES = 2

consistency_output_csv = f"writer_consistency_{TARGET_CHAR_A}_vs_{TARGET_CHAR_B}.csv"
distinctiveness_output_csv = f"writer_distinctiveness_{TARGET_CHAR_A}.csv"
atypical_output_csv = f"atypical_glyphs_{TARGET_CHAR_A}.csv"

# Step 1: Load and Prepare Data 
print("Loading data and creating unified DataFrame...")
data = np.load(embedding_path, allow_pickle=True)
df_main = pd.DataFrame({'path': data["paths"], 'embedding': list(data["embeddings"])})
df_meta = pd.read_csv(train_split_path).append(pd.read_csv(test_split_path))
df_main = pd.merge(df_main, df_meta[['path', 'tm_id', 'base_char_name']], on='path', how='left')
df_main.dropna(inplace=True)
print(f"Loaded and merged data for {len(df_main)} glyphs.")

# WRITER CONSISTENCY (e.g., Epsilon vs. Alpha)
print("\nRunning Analysis 1: Writer Consistency ")

results_consistency = []
all_writers = df_main['tm_id'].unique()

for writer in tqdm(all_writers, desc="Analyzing Writer Consistency"):
    writer_df = df_main[df_main['tm_id'] == writer]

    char_a_embs_list = writer_df[writer_df['base_char_name'] == TARGET_CHAR_A]['embedding'].tolist()
    char_b_embs_list = writer_df[writer_df['base_char_name'] == TARGET_CHAR_B]['embedding'].tolist()

    if len(char_a_embs_list) < MIN_SAMPLES or len(char_b_embs_list) < MIN_SAMPLES:
        continue

    char_a_embs = np.stack(char_a_embs_list)
    char_b_embs = np.stack(char_b_embs_list)

    mean_a = np.mean(char_a_embs, axis=0).reshape(1, -1)
    cohesion_a = np.mean(cosine_similarity(char_a_embs, mean_a))

    mean_b = np.mean(char_b_embs, axis=0).reshape(1, -1)
    distance_a_b = 1 - cosine_similarity(mean_a, mean_b)[0, 0]

    results_consistency.append({
        "Writer_ID": writer,
        f"{TARGET_CHAR_A.capitalize()}_Cohesion": cohesion_a,
        f"{TARGET_CHAR_A.capitalize()}_{TARGET_CHAR_B.capitalize()}_Distance": distance_a_b,
        f"Num_{TARGET_CHAR_A.capitalize()}": len(char_a_embs),
        f"Num_{TARGET_CHAR_B.capitalize()}": len(char_b_embs)
    })

if results_consistency:
    df_consistency = pd.DataFrame(results_consistency).sort_values(
        by=f"{TARGET_CHAR_A.capitalize()}_{TARGET_CHAR_B.capitalize()}_Distance", ascending=False)
    df_consistency.to_csv(consistency_output_csv, index=False)
    print(f"Consistency analysis complete. Saved to '{consistency_output_csv}'")
    print("Top 5 Most Distinct Writers (Epsilon vs Alpha) ")
    print(df_consistency.head(5).to_string())
else:
    print(f"Could not find enough data to compare '{TARGET_CHAR_A}' and '{TARGET_CHAR_B}'.")

#  ANALYSIS 2: WRITER & GLYPH DISTINCTIVENESS (e.g., for Epsilon)
print("\nRunning Analysis 2: Character Distinctiveness ")

target_char_df = df_main[df_main['base_char_name'] == TARGET_CHAR_A]
if not target_char_df.empty:
    all_target_embs = np.stack(target_char_df['embedding'].values)
    global_mean_vector = np.mean(all_target_embs, axis=0)

    scores = []
    for emb in target_char_df['embedding']:
        distance = 1 - cosine_similarity(emb.reshape(1, -1), global_mean_vector.reshape(1, -1))[0, 0]
        scores.append(distance)

    target_char_df = target_char_df.copy()
    target_char_df['Distinctiveness_Score'] = scores

    atypical_glyphs_df = target_char_df.sort_values(by="Distinctiveness_Score", ascending=False)
    atypical_glyphs_df[['path', 'tm_id', 'Distinctiveness_Score']].to_csv(atypical_output_csv, index=False)
    print(f"\nAtypical glyph analysis complete. Saved to '{atypical_output_csv}'")
    print(f"\nTop 10 Most Atypical '{TARGET_CHAR_A.capitalize()}' Glyphs ")
    print(atypical_glyphs_df[['path', 'tm_id', 'Distinctiveness_Score']].head(10).to_string())

    # Group by writer ID and calculate the mean distinctiveness
    writer_avg_distinctiveness_series = atypical_glyphs_df.groupby('tm_id')['Distinctiveness_Score'].mean().sort_values(
        ascending=False)

    # Convert the resulting Series to a DataFrame to have control over column names
    df_distinctiveness = writer_avg_distinctiveness_series.reset_index()
    # Rename the columns to a standard format that the plotting script can expect
    df_distinctiveness.columns = ['Writer_ID', 'Distinctiveness_Score']

    # Save the clean DataFrame to the CSV
    df_distinctiveness.to_csv(distinctiveness_output_csv, index=False)

    print(f"\nWriter distinctiveness analysis complete. Saved to '{distinctiveness_output_csv}'")
    print(f"\nTop 5 Writers with Most Atypical '{TARGET_CHAR_A.capitalize()}'s (on average) ")
    # Print from the new DataFrame for consistency
    print(df_distinctiveness.head(5).to_string())
else:
    print(f"No data found for the target character: '{TARGET_CHAR_A}'")