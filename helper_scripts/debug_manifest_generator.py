# debug_manifest_generator.py
import argparse
import json
from pathlib import Path
from collections import defaultdict
import random

# --- NOTE: Copy the EXACT 'create_char_normalization_map' function from your
# --- 'create_full_manifest.py' file and paste it here.
# --- This is critical for debugging the REAL logic.
def create_char_normalization_map(json_path):
    """
    Creates a robust, direct map from every category ID to a normalized
    base character name using a comprehensive variant map.
    """
    with open(json_path) as f:
        categories = json.load(f)['categories']

    VARIANT_MAP = {
        'α': 'alpha', 'Α': 'alpha', 'ᾶ': 'alpha', 'Ά': 'alpha', 'ά': 'alpha', 'ἀ': 'alpha', 'Ἀ': 'alpha',
        'ἂ': 'alpha', 'ἃ': 'alpha', 'ἄ': 'alpha', 'Ἄ': 'alpha', 'ἆ': 'alpha', 'Ἆ': 'alpha', 'ᾱ': 'alpha',
        'Ᾱ': 'alpha', 'ἁ': 'alpha', 'Ἁ': 'alpha', 'a': 'alpha', 'A': 'alpha', 'β': 'beta', 'Β': 'beta',
        'γ': 'gamma', 'Γ': 'gamma', 'g': 'gamma', 'δ': 'delta', 'Δ': 'delta', 'ε': 'epsilon', 'Ε': 'epsilon',
        'έ': 'epsilon', 'Έ': 'epsilon', 'ἐ': 'epsilon', 'Ἐ': 'epsilon', 'ἒ': 'epsilon', 'Ἕ': 'epsilon',
        'ἕ': 'epsilon', 'Ἑ': 'epsilon', 'ζ': 'zeta', 'Ζ': 'zeta', 'η': 'eta', 'Η': 'eta', 'ή': 'eta',
        'Ή': 'eta', 'ἠ': 'eta', 'Ἠ': 'eta', 'ἢ': 'eta', 'ἣ': 'eta', 'ἥ': 'eta', 'Ἥ': 'eta', 'ἧ': 'eta',
        'Ἧ': 'eta', 'ῆ': 'eta', 'θ': 'theta', 'Θ': 'theta', 'ι': 'iota', 'Ι': 'iota', 'ί': 'iota',
        'Ί': 'iota', 'ἰ': 'iota', 'Ἰ': 'iota', 'ἒ': 'iota', 'ἳ': 'iota', 'ἴ': 'iota', 'Ἴ': 'iota',
        'ἶ': 'iota', 'Ἶ': 'iota', 'ἱ': 'iota', 'Ἱ': 'iota', 'ἷ': 'iota', 'ῒ': 'iota', 'ῗ': 'iota',
        'ϊ': 'iota', 'ΐ': 'iota', 'Ϊ': 'iota', 'i': 'iota', 'I': 'iota', 'κ': 'kappa', 'Κ': 'kappa',
        'k': 'kappa', 'λ': 'lambda', 'Λ': 'lambda', 'μ': 'mu', 'Μ': 'mu', 'm': 'mu', 'ν': 'nu',
        'Ν': 'nu', 'n': 'nu', 'ξ': 'xi', 'Ξ': 'xi', 'ο': 'omicron', 'Ο': 'omicron', 'ό': 'omicron',
        'Ό': 'omicron', 'ὀ': 'omicron', 'Ὀ': 'omicron', 'ὃ': 'omicron', 'ὄ': 'omicron', 'Ὄ': 'omicron',
        'π': 'pi', 'Π': 'pi', 'ρ': 'rho', 'Ρ': 'rho', 'ῥ': 'rho', 'Ῥ': 'rho', 'σ': 'sigma', 'ς': 'sigma',
        'Σ': 'sigma', 'ϲ': 'sigma', 'Ϲ': 'sigma', 's': 'sigma', 'c': 'sigma', 'τ': 'tau', 'Τ': 'tau',
        'υ': 'upsilon', 'Υ': 'upsilon', 'ύ': 'upsilon', 'Ύ': 'upsilon', 'ϋ': 'upsilon', 'Ϋ': 'upsilon',
        'ὐ': 'upsilon', 'ὒ': 'upsilon', 'ὔ': 'upsilon', 'ὖ': 'upsilon', 'ὑ': 'upsilon', 'Ὑ': 'upsilon',
        'ᓓ': 'upsilon', 'ὕ': 'upsilon', 'Ὕ': 'upsilon', 'ὗ': 'upsilon', 'φ': 'phi', 'Φ': 'phi',
        'χ': 'chi', 'Χ': 'chi', 'ψ': 'psi', 'Ψ': 'psi', 'ω': 'omega', 'Ω': 'omega', 'ώ': 'omega',
        'Ώ': 'omega', 'ὠ': 'omega', 'Ὠ': 'omega', 'ὢ': 'omega', 'ὥ': 'omega', 'ὦ': 'omega',
        'ὧ': 'omega', 'ῶ': 'omega',
    }
    id_to_base_char = {}
    for cat in categories:
        char_name = VARIANT_MAP.get(cat['name'])
        if char_name:
            id_to_base_char[cat['id']] = char_name
    return id_to_base_char
# --- End of copied function ---

def generate_debug_report(args):
    merged_dir = Path(args.merged_dir)
    metadata_json = Path(args.metadata_json)
    output_html = Path("debug_report.html")

    print("Loading the EXACT character normalization map from your script...")
    id_to_char_name_map = create_char_normalization_map(metadata_json)
    if not id_to_char_name_map:
        print("ERROR: Character map is empty! Check the logic in 'create_char_normalization_map'.")
        return

    print(f"Scanning '{merged_dir}' to collect glyphs by assigned category...")
    glyphs_by_category = defaultdict(list)

    # We only need a sample, not all glyphs, so this will be fast
    all_glyph_files = list(merged_dir.glob("**/*.jpg"))
    random.shuffle(all_glyph_files)

    for glyph_file in all_glyph_files:
        try:
            parts = glyph_file.stem.split('_')
            cat_id_str = parts[-2].replace('cat', '')
            category_id = int(cat_id_str)

            # Only add to our list if it's a valid ID and we need more samples
            if category_id in id_to_char_name_map and len(glyphs_by_category[category_id]) < 20:
                glyphs_by_category[category_id].append(glyph_file.as_posix())

        except (IndexError, ValueError):
            continue

    print("Generating HTML report...")
    with open(output_html, "w") as f:
        f.write("<html><head><title>Manifest Debug Report</title>")
        f.write("<style>body {font-family: sans-serif;} h2 {background-color: #f0f0f0; padding: 5px;} ")
        f.write("img {border: 1px solid #ccc; margin: 2px; width: 64px; height: 64px;}</style>")
        f.write("</head><body><h1>Manifest Labeling Sanity Check</h1>")

        for cat_id in sorted(glyphs_by_category.keys()):
            char_name = id_to_char_name_map.get(cat_id, "Unknown")
            f.write(f"<h2>Category ID: {cat_id} &rarr; Mapped to: '{char_name.upper()}'</h2>")
            for img_path in glyphs_by_category[cat_id]:
                f.write(f'<img src="{img_path}" title="{Path(img_path).name}">')
            f.write("<hr>")

        f.write("</body></html>")

    print(f"\nReport generated! Open the file 'debug_report.html' in your browser.")
    print("Look for sections where the images DO NOT match the mapped character name.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a visual debug report for the manifest creation logic.")
    parser.add_argument("--merged_dir", type=str, required=True, help="Path to the directory with organized glyphs (e.g., 'pipeline_output/4_sanitized_glyphs').")
    parser.add_argument("--metadata_json", type=str, required=True, help="Path to the metadata JSON file.")
    args = parser.parse_args()
    generate_debug_report(args)