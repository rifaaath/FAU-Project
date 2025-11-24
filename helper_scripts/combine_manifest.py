# combine_manifests.py
import pandas as pd

# --- CONFIGURE YOUR PATHS HERE ---
TRAIN_MANIFEST_PATH = "pipeline_output/6_final_splits/train_writer_disjoint.csv"
TEST_MANIFEST_PATH = "pipeline_output/6_final_splits/test_writer_disjoint.csv"
OUTPUT_COMBINED_PATH = "pipeline_output/combined_manifest_for_embedding.csv"
# --------------------------------

print(f"Loading training manifest from: {TRAIN_MANIFEST_PATH}")
df_train = pd.read_csv(TRAIN_MANIFEST_PATH)

print(f"Loading test manifest from: {TEST_MANIFEST_PATH}")
df_test = pd.read_csv(TEST_MANIFEST_PATH)

df_combined = pd.concat([df_train, df_test], ignore_index=True)

df_combined.to_csv(OUTPUT_COMBINED_PATH, index=False)

print(f"\nSuccessfully combined manifests.")
print(f"   Training glyphs: {len(df_train)}")
print(f"   Testing glyphs:  {len(df_test)}")
print(f"   Total glyphs:    {len(df_combined)}")
print(f"Combined manifest saved to: {OUTPUT_COMBINED_PATH}")