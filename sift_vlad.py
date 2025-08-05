import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import joblib
from unidecode import unidecode
from sklearn.model_selection import GroupShuffleSplit
import json

# --- Config ---
embedding_path = "page_indep_embeddings.npz"
full_manifest_path = "glyph_manifest_full.csv"
json_path = "papytwin/HomerCompTraining/HomerCompTraining.json"
N_CLUSTERS_PER_CHAR = 16  # <-- Let's reduce this to a safer number first
output_descriptors_path = "page_descriptors_class_vlad.npz"
POWER_NORM=0.5
# --- End Config ---

# --- Step 1: Create a single, unified DataFrame ---
print("Loading data and creating a unified DataFrame...")
data = np.load(embedding_path, allow_pickle=True)
df_main = pd.DataFrame({'path': data["paths"], 'embedding': list(data["embeddings"])})
df_meta = pd.read_csv(full_manifest_path)
df_main = pd.merge(df_main, df_meta[['path', 'tm_id', 'category_id']], on='path', how='left')


def get_doc_id(path):
    parts = Path(path).stem.split("_");
    return f"{parts[0]}_{parts[1]}" if len(parts) > 2 else Path(path).stem


df_main['doc_id'] = df_main['path'].apply(get_doc_id)
df_main.dropna(inplace=True)

# --- Step 2: Normalize Character Labels ---
print("Normalizing character labels...")
with open(json_path) as f:
    categories = json.load(f)['categories']
id_to_base_char = {cat['id']: unidecode(cat['name']).lower() for cat in categories}
base_chars = sorted(list(set(id_to_base_char.values())))
base_char_to_id = {name: i for i, name in enumerate(base_chars)}
normalization_map = {spec_id: base_char_to_id[base_name] for spec_id, base_name in id_to_base_char.items()}
df_main['base_char_id'] = df_main['category_id'].map(normalization_map)
df_main.dropna(subset=['base_char_id'], inplace=True)
df_main['base_char_id'] = df_main['base_char_id'].astype(int)

# --- Step 3: Perform Page-Independent Split ---
print("Performing page-independent split...")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(splitter.split(df_main, groups=df_main['doc_id']))
df_train = df_main.iloc[train_indices]

# --- Step 4: Train k-means Codebook for Each Character Class ---
print("Training a k-means codebook for each character class...")
character_codebooks = {}
# Use the new 'base_char_id' for grouping
train_char_ids = df_train['base_char_id'].unique()

for base_char_id in tqdm(train_char_ids, desc="Training Character Codebooks"):
    # --- ✅ SIMPLIFIED AND CORRECTED DATA SELECTION ---
    # Select rows for the current character directly from the training dataframe
    char_train_df = df_train[df_train['base_char_id'] == base_char_id]

    if len(char_train_df) < N_CLUSTERS_PER_CHAR:
        # print(f"Skipping char ID {base_char_id}, not enough samples: {len(char_train_df)}")
        continue

    char_train_embs = np.stack(char_train_df['embedding'].values)

    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS_PER_CHAR, random_state=42, n_init=3)
    kmeans.fit(char_train_embs)
    character_codebooks[base_char_id] = kmeans

print(f"Successfully trained {len(character_codebooks)} character-specific codebooks.")
if not character_codebooks:
    print(
        "❌ FATAL: Could not train any codebooks. The number of samples per character in the training set may be too low.")
    exit()

# --- Step 5: Create Class-Specific VLAD Descriptors ---
print("Creating class-specific VLAD descriptors for all document pages...")
doc_char_embs = {
    doc_id: {char_id: np.stack(group['embedding'].values) for char_id, group in doc_group.groupby('base_char_id')} for
    doc_id, doc_group in df_main.groupby('doc_id')}

final_page_descriptors = {}
valid_char_ids = sorted(list(character_codebooks.keys()))

for doc_id, char_groups in tqdm(doc_char_embs.items(), desc="Calculating Class VLADs"):
    doc_final_vlad = []
    for base_char_id in valid_char_ids:
        kmeans = character_codebooks[base_char_id]
        cluster_centers = kmeans.cluster_centers_
        embs_np = char_groups.get(base_char_id, np.array([]))

        vlad_vec = np.zeros_like(cluster_centers)
        if len(embs_np) > 0:
            predictions = kmeans.predict(embs_np)
            for i in range(N_CLUSTERS_PER_CHAR):
                assigned_mask = (predictions == i)
                if np.any(assigned_mask):
                    residual = np.sum(embs_np[assigned_mask] - cluster_centers[i], axis=0)
                    vlad_vec[i, :] = residual

        vlad_vec = vlad_vec.flatten()
        vlad_vec = np.sign(vlad_vec) * (np.abs(vlad_vec) ** POWER_NORM)
        vlad_vec = normalize(vlad_vec.reshape(1, -1), norm='l2').flatten()
        doc_final_vlad.append(vlad_vec)

    if doc_final_vlad:
        final_page_descriptors[doc_id] = np.concatenate(doc_final_vlad)

# --- Step 6: Save the Final Descriptors ---
all_doc_ids = list(final_page_descriptors.keys())
all_descriptors = np.array([final_page_descriptors[did] for did in all_doc_ids])

np.savez_compressed(
    output_descriptors_path,
    descriptors=all_descriptors,
    doc_ids=np.array(all_doc_ids)
)
print(f"\n✅ Success! Saved {len(all_descriptors)} class-specific VLAD page descriptors to '{output_descriptors_path}'")