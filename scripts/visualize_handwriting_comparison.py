import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random


def generate_comparison_image(args):
    """
    Generates a visual comparison grid of glyphs for the specified writers.
    """
    writer_ids = args.writer_ids
    glyph_root_dir = Path(args.glyph_dir)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Visualization Parameters 
    GLYPHS_PER_ROW = 15
    GLYPH_SIZE = (80, 80)  # Resize glyphs for a consistent grid
    PADDING = 10
    HEADER_HEIGHT = 50
    LABEL_WIDTH = 150
    BG_COLOR = (255, 255, 255)

    try:
        FONT_LARGE = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        FONT_MEDIUM = ImageFont.truetype("DejaVuSans.ttf", 18)
    except IOError:
        print("Warning: DejaVuSans font not found. Using default font.")
        FONT_LARGE = ImageFont.load_default()
        FONT_MEDIUM = ImageFont.load_default()

    # Collect Glyphs for Each Writer 
    writer_glyphs = {}
    print("Collecting glyphs for specified writers...")
    for writer_id in writer_ids:
        writer_folder = glyph_root_dir / writer_id
        if not writer_folder.exists():
            print(f"Warning: Directory for writer {writer_id} not found. Skipping.")
            continue

        # Find all glyphs, optionally filtering by character name
        glob_pattern = f"*_cat{args.char_cat_id}_*.jpg" if args.char_cat_id else "*.jpg"
        glyph_paths = list(writer_folder.glob(glob_pattern))

        if not glyph_paths:
            print(f"Warning: No matching glyphs found for writer {writer_id} (Filter: {glob_pattern}). Skipping.")
            continue

        # Shuffle and take a sample
        random.shuffle(glyph_paths)
        writer_glyphs[writer_id] = glyph_paths[:GLYPHS_PER_ROW]
        print(f"  - Found {len(glyph_paths)} glyphs for {writer_id}, sampling {len(writer_glyphs[writer_id])}.")

    if not writer_glyphs:
        print("Error: No glyphs found for any of the specified writers. Aborting.")
        return

    # Calculate Canvas Size and Create Image 
    num_writers = len(writer_glyphs)
    row_height = GLYPH_SIZE[1] + PADDING
    canvas_width = LABEL_WIDTH + (GLYPHS_PER_ROW * (GLYPH_SIZE[0] + PADDING))
    canvas_height = HEADER_HEIGHT + (num_writers * row_height)

    canvas = Image.new("RGB", (canvas_width, canvas_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Draw title
    title = f"Handwriting Comparison: {' vs. '.join(writer_glyphs.keys())}"
    if args.char_cat_id:
        title += f" (Character ID: {args.char_cat_id})"
    draw.text((PADDING, PADDING), title, font=FONT_LARGE, fill=(0, 0, 0))

    # Draw Glyphs for Each Writer 
    current_y = HEADER_HEIGHT
    for writer_id, glyph_paths in writer_glyphs.items():
        # Draw writer ID label
        draw.text((PADDING, current_y + GLYPH_SIZE[1] // 2 - 10), writer_id, font=FONT_MEDIUM, fill=(50, 50, 50))

        # Draw the glyphs in a row
        for i, glyph_path in enumerate(glyph_paths):
            try:
                with Image.open(glyph_path) as img:
                    img = img.resize(GLYPH_SIZE, Image.LANCZOS)
                    # Paste glyph onto the canvas
                    paste_x = LABEL_WIDTH + (i * (GLYPH_SIZE[0] + PADDING))
                    canvas.paste(img, (paste_x, current_y))
            except Exception as e:
                print(f"Warning: Could not process glyph {glyph_path.name}. Error: {e}")

        current_y += row_height

    # Save the final image
    canvas.save(output_path)
    print(f"\nComparison image saved to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a visual comparison of handwriting for multiple writers.")
    parser.add_argument(
        "writer_ids",
        nargs='+',
        help="One or more writer IDs to compare (e.g., TM_60306 TM_61022)."
    )
    parser.add_argument(
        "--glyph_dir",
        type=str,
        default="glyph_crops_yolo_organized_by_tm",  # Point to your organized glyphs
        help="The root directory where glyphs are organized into subfolders by TM_ID."
    )
    parser.add_argument(
        "--char_cat_id",
        type=int,
        default=None,
        help="Optional: Filter to show only glyphs of a specific category ID (e.g., 7 for Epsilon)."
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default="handwriting_comparison.png",
        help="The path to save the output image."
    )
    args = parser.parse_args()
    generate_comparison_image(args)