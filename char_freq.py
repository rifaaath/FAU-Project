import json
from collections import Counter
from tqdm import tqdm

# --- CONFIG ---
# Path to your main dataset JSON file
training_json_path = "papytwin/HomerCompTraining/HomerCompTraining.json"
# How many top characters to display in the final report
TOP_N_TO_SHOW = 24  # Set to 24 to see all Greek letters
# --- END CONFIG ---

print(f"Loading dataset metadata from: {training_json_path}")
with open(training_json_path) as f:
    data = json.load(f)

# --- Step 1: Create a map from category ID to character name ---
if 'categories' not in data:
    print("❌ Error: 'categories' key not found in JSON file.")
    exit()

id_to_char = {cat['id']: cat['name'] for cat in data['categories']}
print(f"Found {len(id_to_char)} unique character categories.")

# --- Step 2: Count all annotations ---
if 'annotations' not in data:
    print("❌ Error: 'annotations' key not found in JSON file.")
    exit()

annotations = data['annotations']
total_annotations = len(annotations)
print(f"Counting {total_annotations} total annotations...")

# Use a Counter for efficient tallying
category_counts = Counter()
for ann in tqdm(annotations, desc="Processing annotations"):
    category_id = ann.get('category_id')
    if category_id is not None:
        category_counts[category_id] += 1

# --- Step 3: Print the ranked report ---
print("\n--- Character Frequency Report ---")
print(f"{'Rank':<5} {'Char':<5} {'ID':<5} {'Count':>10} {'Percentage':>12}")
print("-" * 40)

# Sort the counter by count, descending
for i, (cat_id, count) in enumerate(category_counts.most_common(TOP_N_TO_SHOW), 1):
    char_name = id_to_char.get(cat_id, '???')
    percentage = (count / total_annotations) * 100

    print(f"{i:<5} {char_name:<5} {cat_id:<5} {count:>10,} {percentage:>11.2f}%")

print("-" * 40)

most_common_id = category_counts.most_common(1)[0][0]
most_common_char = id_to_char[most_common_id]
print(f"\n✅ Most frequent character is '{most_common_char}' (ID: {most_common_id}).")