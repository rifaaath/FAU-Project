# visualize_vlad_codebook_v3.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit

# --- Config ---
embedding_path = "embeddings_page_independent.npz"
full_manifest_path = "glyph_manifest_full.csv"  # The original manifest with category IDs

# --- Character to Visualize ---
# Use the MOST frequent character IDs for Epsilon (ε, Ε)
TARGET_CHAR_IDS = [103, 23]
TARGET_CHAR_NAME = "Epsilon"

# --- VLAD/Visualization Parameters ---
N_CLUSTERS = 16
TSNE_PERPLEXITY = 30
TSNE_SAMPLES = 4000
output_plot_path = f"vlad_codebook_visualization_{TARGET_CHAR_NAME}.png"
# --- End Config ---

# --- Step 1: Create a single, unified DataFrame as the source of truth ---
print("Loading data and creating a unified DataFrame...")
data = np.load(embedding_path, allow_pickle=True)
embeddings = data["embeddings"]
paths_from_npz = data["paths"]

# Create a DataFrame from the embeddings that ACTUALLY EXIST
df_main = pd.DataFrame({
    'path': paths_from_npz,
    'embedding': list(embeddings)
})

# Load the full metadata
df_meta = pd.read_csv(full_manifest_path)

# Merge the metadata onto our embedding DataFrame
# This ensures we only have metadata for embeddings that exist.
df_main = pd.merge(df_main, df_meta[['path', 'tm_id', 'category_id']], on='path', how='left')


# --- Step 2: Create Page-Independent Grouping Key and Split ---
def get_doc_id(path):
    parts = Path(path).stem.split("_")
    if len(parts) > 2: return f"{parts[0]}_{parts[1]}"
    return Path(path).stem


df_main['doc_id'] = df_main['path'].apply(get_doc_id)
df_main.dropna(inplace=True)  # Remove rows that failed to merge or get a doc_id

print(f"Created unified DataFrame with {len(df_main)} valid entries.")

print("Performing page-independent split...")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(splitter.split(df_main, groups=df_main['doc_id']))
df_train = df_main.iloc[train_indices]

# --- Step 3: Isolate the target character embeddings FROM THE TRAINING SET ---
print(f"Isolating '{TARGET_CHAR_NAME}' glyphs from the training set...")
# Select rows for the current character directly from the training dataframe
char_train_df = df_train[df_train['category_id'].isin(TARGET_CHAR_IDS)]

if char_train_df.empty:
    print(f"❌ FATAL Error: No glyphs found for character IDs {TARGET_CHAR_IDS} in the training set.")
    print("This means that, by chance, the page-independent split put all Epsilons in the test set.")
    print("Try a different character or a different random_state in the splitter.")
    exit()

char_train_embs = np.stack(char_train_df['embedding'].values)
print(f"Found {len(char_train_embs)} training samples for '{TARGET_CHAR_NAME}'.")

# --- Step 4: Train k-means on these embeddings ---
if len(char_train_embs) < N_CLUSTERS:
    print(f"❌ Error: Not enough samples ({len(char_train_embs)}) to train k-means with k={N_CLUSTERS}.")
    exit()

print(f"Training k-means with {N_CLUSTERS} clusters...")
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=3)
cluster_assignments = kmeans.fit_predict(char_train_embs)
cluster_centers = kmeans.cluster_centers_

# --- Step 5: Run t-SNE and Plot ---
# (This section is the same as the previous script and should now work)
if len(char_train_embs) > TSNE_SAMPLES:
    print(f"Running t-SNE on a random subset of {TSNE_SAMPLES} samples...")
    indices = np.random.choice(len(char_train_embs), TSNE_SAMPLES, replace=False)
    embs_for_tsne = char_train_embs[indices]
    labels_for_tsne = cluster_assignments[indices]
else:
    print("Running t-SNE on all samples...")
    embs_for_tsne = char_train_embs
    labels_for_tsne = cluster_assignments

tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=42, init='pca')
embs_2d = tsne.fit_transform(embs_for_tsne)

centers_2d = []
for i in range(N_CLUSTERS):
    center = cluster_centers[i]
    distances = np.linalg.norm(embs_for_tsne - center, axis=1)
    closest_emb_idx = np.argmin(distances)
    centers_2d.append(embs_2d[closest_emb_idx])
centers_2d = np.array(centers_2d)

print("Generating plot...")
plt.figure(figsize=(12, 10))
palette = sns.color_palette("husl", N_CLUSTERS)
sns.scatterplot(x=embs_2d[:, 0], y=embs_2d[:, 1], hue=labels_for_tsne, palette=palette, legend=False, s=10, alpha=0.6)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='X', s=100, c='black', label='k-means Cluster Centers')
plt.title(f"t-SNE Visualization of k-means Clusters for '{TARGET_CHAR_NAME}' Glyphs (k={N_CLUSTERS})")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.tight_layout()
plt.savefig(output_plot_path)
print(f"\n✅ Visualization saved to '{output_plot_path}'")