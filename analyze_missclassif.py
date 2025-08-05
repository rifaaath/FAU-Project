import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm

# --- Config ---
embedding_path = "embeddings.npz"
output_csv_path = "misclassified.csv"
output_plot_path = "misclassification_plot.png"
top_n_to_plot = 15  # How many top errors to show in the plot

# --- Load Data ---
print("Loading data for analysis...")
data = np.load(embedding_path)
X = data["embeddings"]
paths = data["paths"]

# --- Use the EXACT SAME split as in training ---
writer_ids = np.array([p.split("/")[-2] for p in paths])
unique_writers = np.unique(writer_ids)
np.random.seed(42)  # Use the same seed to get the same shuffle
np.random.shuffle(unique_writers)

test_writer_ids = unique_writers[int(0.8 * len(unique_writers)):]
print(f"Analyzing misclassifications on {len(test_writer_ids)} unseen writers.")

# --- Create test dataset from embeddings ---
test_mask = np.isin(writer_ids, test_writer_ids)
X_test = X[test_mask]
writer_ids_test = writer_ids[test_mask]
paths_test = paths[test_mask]

# --- Use separate lists for embeddings and paths ---
writer_embs_map = defaultdict(list)
writer_paths_map = defaultdict(list)
for emb, wid, path in zip(X_test, writer_ids_test, paths_test):
    writer_embs_map[wid].append(emb)
    writer_paths_map[wid].append(path)

# --- Build Writer Prototypes (The "Gallery") ---
print("Building writer prototypes from the test set...")
gallery_writers = []
gallery_vectors = []
for wid, embs in writer_embs_map.items():
    gallery_writers.append(wid)
    gallery_vectors.append(np.mean(embs, axis=0))

gallery_vectors = np.array(gallery_vectors)
gallery_writer_map = {writer: i for i, writer in enumerate(gallery_writers)}

# --- Evaluate and Log Misclassifications ---
print("Evaluating retrieval and logging errors...")
misclassifications = []
correct_predictions = 0
total_queries = 0

for true_wid, query_embs in tqdm(writer_embs_map.items(), desc="Evaluating writers"):
    query_paths = writer_paths_map[true_wid]

    if len(query_embs) < 2:
        continue

    for i in range(len(query_embs)):
        query_emb = query_embs[i]
        query_path = query_paths[i]

        temp_gallery_vectors = gallery_vectors.copy()
        other_embs = np.delete(query_embs, i, axis=0)

        writer_idx_in_gallery = gallery_writer_map[true_wid]
        temp_gallery_vectors[writer_idx_in_gallery] = np.mean(other_embs, axis=0)

        similarities = cosine_similarity(query_emb.reshape(1, -1), temp_gallery_vectors)[0]

        predicted_idx = np.argmax(similarities)
        predicted_wid = gallery_writers[predicted_idx]

        total_queries += 1
        if predicted_wid == true_wid:
            correct_predictions += 1
        else:
            true_writer_idx = gallery_writer_map[true_wid]
            similarity_to_true_writer = similarities[true_writer_idx]
            similarity_to_predicted_writer = similarities[predicted_idx]

            misclassifications.append({
                "Query_Glyph_Path": query_path,
                "True_Writer_ID": true_wid,
                "Predicted_Writer_ID": predicted_wid,
                "Similarity_to_True": similarity_to_true_writer,
                "Similarity_to_Predicted": similarity_to_predicted_writer,
                "Similarity_Difference": similarity_to_predicted_writer - similarity_to_true_writer
            })

# --- Final Accuracy Result ---
retrieval_accuracy = (correct_predictions / total_queries) * 100 if total_queries > 0 else 0
print(f"\n✅ Top-1 Writer-Disjoint Retrieval Accuracy: {retrieval_accuracy:.2f}%")

# --- Save and Plot Misclassifications ---
if misclassifications:
    df_misclassified = pd.DataFrame(misclassifications)
    df_misclassified.to_csv(output_csv_path, index=False)
    print(f"✅ Saved {len(df_misclassified)} misclassifications to {output_csv_path}")

    # --- ✅ NEW PLOTTING SECTION ---
    print(f"Generating plot of top {top_n_to_plot} errors...")

    # Create a confusion pair string for easy counting
    df_misclassified['Confusion_Pair'] = df_misclassified.apply(
        lambda row: f"{row['True_Writer_ID']} -> {row['Predicted_Writer_ID']}", axis=1
    )

    # Count the occurrences of each confusion pair
    confusion_counts = df_misclassified['Confusion_Pair'].value_counts().nlargest(top_n_to_plot)

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x=confusion_counts.values, y=confusion_counts.index, palette="viridis")

    plt.title(f'Top {top_n_to_plot} Most Frequent Writer Misclassifications', fontsize=16)
    plt.xlabel('Number of Misclassified Glyphs', fontsize=12)
    plt.ylabel('Confusion (True Writer -> Predicted Writer)', fontsize=12)
    plt.tight_layout()  # Adjust layout to make sure everything fits

    plt.savefig(output_plot_path)
    print(f"✅ Plot saved to {output_plot_path}")

else:
    print("🎉 No misclassifications found!")