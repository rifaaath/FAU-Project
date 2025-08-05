# create_page_independent_split.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# --- Config ---
manifest_path = "glyph_manifest_full.csv" # Your full, correct manifest
output_dir = Path("page_independent_splits")
output_dir.mkdir(exist_ok=True)
# --- End Config ---

print(f"Loading manifest from {manifest_path}...")
df = pd.read_csv(manifest_path)

# --- Create a unique document ID for each glyph ---
# The document ID is the filename part before the page/frame number
# e.g., 'p_18177_r_3_001.jpg' -> 'p_18177'
def get_doc_id(path):
    # This logic may need adjustment based on the most common filename format
    # A simple approach is to take the first two parts of the filename
    parts = Path(path).stem.split("_")
    if len(parts) > 2:
        return f"{parts[0]}_{parts[1]}"
    return Path(path).stem

df['doc_id'] = df['path'].apply(get_doc_id)

print(f"Found {len(df['doc_id'].unique())} unique documents.")
print(f"Found {len(df['tm_id'].unique())} unique writers.")

# --- Perform the Group Shuffle Split ---
# This ensures that all glyphs with the same 'doc_id' stay in the same set.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(splitter.split(df, groups=df['doc_id']))

df_train = df.iloc[train_indices]
df_test = df.iloc[test_indices]

# --- Save the splits to CSV files ---
train_path = output_dir / "train_split.csv"
test_path = output_dir / "test_split.csv"

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print("\n--- Split Summary ---")
print(f"Total Glyphs: {len(df)}")
print(f"Training Glyphs: {len(df_train)} ({len(df_train['doc_id'].unique())} docs, {len(df_train['tm_id'].unique())} writers)")
print(f"Testing Glyphs: {len(df_test)} ({len(df_test['doc_id'].unique())} docs, {len(df_test['tm_id'].unique())} writers)")
print(f"\n✅ Page-independent splits saved to '{output_dir.resolve()}'")