# visualize_retrieval_examples.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# --- Config ---
embedding_path = "embeddings_page_independent.npz"
test_split_path = "final_splits/test_final.csv"

# --- Visualization Parameters ---
NUM_EXAMPLES_TO_VISUALIZE = 10  # How many random queries to save images for
TOP_K_TO_SHOW = 5  # Show the top 5 results for each query
IMAGE_SIZE = (100, 100)  # Resize glyphs for consistent display

output_dir = Path("retrieval_visualizations")
output_dir.mkdir(exist_ok=True)
# --- End Config ---

# --- Font and Color Config ---
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    FONT_SMALL = ImageFont.truetype("DejaVuSans.ttf", 12)
except IOError:
    print("Warning: DejaVuSans font not found. Using default font.")
    FONT = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()
CORRECT_COLOR = (0, 150, 0)  # Dark Green
INCORRECT_COLOR = (200, 0, 0)  # Red
BG_COLOR = (255, 255, 255)

# --- Step 1: Load and Prepare Data ---
print("Loading data...")
data = np.load(embedding_path, allow_pickle=True)
path_to_embedding = {path: emb for path, emb in zip(data["paths"], data["embeddings"])}
df_test = pd.read_csv(test_split_path)

gallery_embs = []
gallery_writers = []
gallery_paths = []
for _, row in df_test.iterrows():
    if row['path'] in path_to_embedding:
        gallery_embs.append(path_to_embedding[row['path']])
        gallery_writers.append(row['tm_id'])
        gallery_paths.append(row['path'])

gallery_embs = np.array(gallery_embs)
gallery_writers = np.array(gallery_writers)
gallery_paths = np.array(gallery_paths)
print(f"Gallery contains {len(gallery_embs)} glyphs from the test set.")

# --- Step 2: Select Random Queries to Visualize ---
# Ensure we pick queries from writers who have at least TOP_K_TO_SHOW+1 samples
# so the leave-one-out evaluation is meaningful.
writer_counts = pd.Series(gallery_writers).value_counts()
valid_writers = writer_counts[writer_counts > TOP_K_TO_SHOW].index
valid_query_indices = [i for i, writer in enumerate(gallery_writers) if writer in valid_writers]

if not valid_query_indices:
    print(f"❌ Error: Could not find any writers in the test set with more than {TOP_K_TO_SHOW} glyphs.")
    print("Cannot create meaningful visualizations. Try reducing TOP_K_TO_SHOW.")
    exit()

query_indices = np.random.choice(valid_query_indices, size=min(NUM_EXAMPLES_TO_VISUALIZE, len(valid_query_indices)),
                                 replace=False)
print(f"Selected {len(query_indices)} random queries to visualize.")

# --- Step 3: Perform Retrieval and Generate Image for Each Query ---
for query_idx in tqdm(query_indices, desc="Generating visualizations"):
    query_emb = gallery_embs[query_idx]
    query_path = gallery_paths[query_idx]
    true_writer_id = gallery_writers[query_idx]

    # Create a temporary gallery that excludes the query glyph
    temp_gallery_embs = np.delete(gallery_embs, query_idx, axis=0)
    temp_gallery_writers = np.delete(gallery_writers, query_idx)
    temp_gallery_paths = np.delete(gallery_paths, query_idx)

    # Calculate similarities and get top K results
    similarities = cosine_similarity(query_emb.reshape(1, -1), temp_gallery_embs)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    top_k_indices = sorted_indices[:TOP_K_TO_SHOW]
    top_k_paths = temp_gallery_paths[top_k_indices]
    top_k_writers = temp_gallery_writers[top_k_indices]
    top_k_scores = similarities[top_k_indices]

    # --- Create the visualization canvas ---
    canvas_width = IMAGE_SIZE[0] * (TOP_K_TO_SHOW + 2)  # Space for query + results
    canvas_height = IMAGE_SIZE[1] + 80  # Space for labels
    canvas = Image.new("RGB", (canvas_width, canvas_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # --- Draw the Query Glyph ---
    query_img = Image.open(query_path).resize(IMAGE_SIZE)
    canvas.paste(query_img, (20, 40))
    draw.text((20, 10), "Query Glyph", font=FONT, fill=(0, 0, 0))
    draw.text((20, 45 + IMAGE_SIZE[1]), f"Writer: {true_writer_id}", font=FONT_SMALL, fill=(0, 0, 0))

    # Draw a separator
    draw.line([(IMAGE_SIZE[0] + 40, 10), (IMAGE_SIZE[0] + 40, canvas_height - 10)], fill=(200, 200, 200), width=2)

    # --- Draw the Top K Retrieved Glyphs ---
    for i in range(TOP_K_TO_SHOW):
        result_path = top_k_paths[i]
        result_writer = top_k_writers[i]
        result_score = top_k_scores[i]

        is_correct = (result_writer == true_writer_id)
        border_color = CORRECT_COLOR if is_correct else INCORRECT_COLOR

        result_img = Image.open(result_path).resize(IMAGE_SIZE)

        x_pos = IMAGE_SIZE[0] + 80 + (i * IMAGE_SIZE[0])
        y_pos = 40

        # Draw a colored border around the image
        draw.rectangle(
            (x_pos - 2, y_pos - 2, x_pos + IMAGE_SIZE[0] + 2, y_pos + IMAGE_SIZE[1] + 2),
            fill=border_color
        )
        canvas.paste(result_img, (x_pos, y_pos))

        # Draw labels below the image
        draw.text((x_pos, y_pos + IMAGE_SIZE[1] + 5), f"Rank #{i + 1}", font=FONT, fill=(0, 0, 0))
        draw.text((x_pos, y_pos + IMAGE_SIZE[1] + 25), f"Writer: {result_writer}", font=FONT_SMALL, fill=(0, 0, 0))
        draw.text((x_pos, y_pos + IMAGE_SIZE[1] + 40), f"Score: {result_score:.3f}", font=FONT_SMALL,
                  fill=(100, 100, 100))

    # Save the final image
    query_basename = Path(query_path).stem
    output_path = output_dir / f"retrieval_{query_basename}.png"
    canvas.save(output_path)

print(f"\n✅ Visualizations saved to the '{output_dir.resolve()}' directory.")