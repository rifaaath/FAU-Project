# prepare_final_dataset.py (FINAL, with Manual Normalization)
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# --- Config ---
full_manifest_path = "glyph_manifest_full.csv"
json_path = "papytwin/HomerCompTraining/HomerCompTraining.json"
output_dir = Path("final_splits")
output_dir.mkdir(exist_ok=True)
# --- End Config ---

# --- Step 1: Create a Robust Normalization Map ---
print("Creating character normalization map...")
with open(json_path) as f:
    categories = json.load(f)['categories']

# --- ✅ NEW MANUAL MAPPING FOR ROBUSTNESS ---
GREEK_ALPHABET_MAP = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
    'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
    'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
    'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'ς': 'sigma', 'τ': 'tau',
    'υ': 'upsilon', 'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega'
}
# Also map uppercase versions
GREEK_ALPHABET_MAP.update({k.upper(): v for k, v in GREEK_ALPHABET_MAP.items()})

# This function now provides a canonical name for a glyph
def get_base_char_name(char_str):
    # Check against our explicit map first
    if char_str in GREEK_ALPHABET_MAP:
        return GREEK_ALPHABET_MAP[char_str]
    # Handle ligatures or other symbols if necessary, for now return None
    return None

id_to_base_char = {cat['id']: get_base_char_name(cat['name']) for cat in categories}
base_chars = sorted(list(set(b for b in id_to_base_char.values() if b is not None)))
base_char_to_id = {name: i for i, name in enumerate(base_chars)}
normalization_map = {spec_id: base_char_to_id.get(base_name) for spec_id, base_name in id_to_base_char.items()}
print(f"Created map for {len(base_char_to_id)} base characters.")

# --- Step 2: Load and Apply Normalization ---
print(f"Loading full manifest from: {full_manifest_path}")
df = pd.read_csv(full_manifest_path)

print("Applying character normalization...")
df['base_char_id'] = df['category_id'].map(normalization_map)
# Use a reverse map to get the proper name (e.g., 'epsilon')
id_to_base_name_map = {v: k for k, v in base_char_to_id.items()}
df['base_char_name'] = df['base_char_id'].map(id_to_base_name_map)

df.dropna(subset=['base_char_id'], inplace=True)
df['base_char_id'] = df['base_char_id'].astype(int)

# --- The rest of the script is the same (splitting and saving) ---
def get_doc_id(path):
    parts = Path(path).stem.split("_"); return f"{parts[0]}_{parts[1]}" if len(parts) > 2 else Path(path).stem
df['doc_id'] = df['path'].apply(get_doc_id)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(splitter.split(df, groups=df['doc_id']))
df_train = df.iloc[train_indices]
df_test = df.iloc[test_indices]
train_path = output_dir / "train_final.csv"
test_path = output_dir / "test_final.csv"
df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print("\n--- Split Summary ---")
print(f"Training Glyphs: {len(df_train)}")
print(f"Testing Glyphs: {len(df_test)}")
print(f"\n✅ Final, normalized splits saved to '{output_dir.resolve()}'")
print("\n--- Character Counts in Training Set (Top 5) ---")
print(df_train['base_char_name'].value_counts().nlargest(5))