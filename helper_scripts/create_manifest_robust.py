# create_manifest_debug_v2.py
import argparse
import csv
import re
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

VARIANT_MAP = {
    'α': 'alpha', 'Α': 'alpha', 'ᾶ': 'alpha', 'Ά': 'alpha', 'ά': 'alpha', 'ἀ': 'alpha', 'Ἀ': 'alpha', 'ἂ': 'alpha',
    'ἃ': 'alpha', 'ἄ': 'alpha', 'Ἄ': 'alpha', 'ἆ': 'alpha', 'Ἆ': 'alpha', 'ᾱ': 'alpha', 'Ᾱ': 'alpha', 'ἁ': 'alpha',
    'Ἁ': 'alpha', 'a': 'alpha', 'A': 'alpha', 'β': 'beta', 'Β': 'beta', 'γ': 'gamma', 'Γ': 'gamma', 'g': 'gamma',
    'δ': 'delta', 'Δ': 'delta', 'ε': 'epsilon', 'Ε': 'epsilon', 'έ': 'epsilon', 'Έ': 'epsilon', 'ἐ': 'epsilon',
    'Ἐ': 'epsilon', 'ἒ': 'epsilon', 'Ἕ': 'epsilon', 'ἕ': 'epsilon', 'Ἑ': 'epsilon', 'ζ': 'zeta', 'Ζ': 'zeta',
    'η': 'eta', 'Η': 'eta', 'ή': 'eta', 'Ή': 'eta', 'ἠ': 'eta', 'Ἠ': 'eta', 'ἢ': 'eta', 'ἣ': 'eta', 'ἥ': 'eta',
    'Ἥ': 'eta', 'ἧ': 'eta', 'Ἧ': 'eta', 'ῆ': 'eta', 'θ': 'theta', 'Θ': 'theta', 'ι': 'iota', 'Ι': 'iota',
    'ί': 'iota', 'Ί': 'iota', 'ἰ': 'iota', 'Ἰ': 'iota', 'ἳ': 'iota', 'ἴ': 'iota', 'Ἴ': 'iota', 'ἶ': 'iota',
    'Ἶ': 'iota', 'ἱ': 'iota', 'Ἱ': 'iota', 'ἷ': 'iota', 'ῒ': 'iota', 'ῗ': 'iota', 'ϊ': 'iota', 'ΐ': 'iota',
    'Ϊ': 'iota', 'i': 'iota', 'I': 'iota', 'κ': 'kappa', 'Κ': 'kappa', 'k': 'kappa', 'λ': 'lambda', 'Λ': 'lambda',
    'μ': 'mu', 'Μ': 'mu', 'm': 'mu', 'ν': 'nu', 'Ν': 'nu', 'n': 'nu', 'ξ': 'xi', 'Ξ': 'xi', 'ο': 'omicron',
    'Ο': 'omicron', 'ό': 'omicron', 'Ό': 'omicron', 'ὀ': 'omicron', 'Ὀ': 'omicron', 'ὃ': 'omicron', 'ὄ': 'omicron',
    'Ὄ': 'omicron', 'π': 'pi', 'Π': 'pi', 'ρ': 'rho', 'Ρ': 'rho', 'ῥ': 'rho', 'Ῥ': 'rho', 'σ': 'sigma',
    'ς': 'sigma', 'Σ': 'sigma', 'ϲ': 'sigma', 'Ϲ': 'sigma', 's': 'sigma', 'c': 'sigma', 'τ': 'tau', 'Τ': 'tau',
    'υ': 'upsilon', 'Υ': 'upsilon', 'ύ': 'upsilon', 'Ύ': 'upsilon', 'ϋ': 'upsilon', 'Ϋ': 'upsilon', 'ὐ': 'upsilon',
    'ὒ': 'upsilon', 'ὔ': 'upsilon', 'ὖ': 'upsilon', 'ὑ': 'upsilon', 'Ὑ': 'upsilon', 'ὓ': 'upsilon', 'ὕ': 'upsilon',
    'Ὕ': 'upsilon', 'ὗ': 'upsilon', 'φ': 'phi', 'Φ': 'phi', 'χ': 'chi', 'Χ': 'chi', 'ψ': 'psi', 'Ψ': 'psi',
    'ω': 'omega', 'Ω': 'omega', 'ώ': 'omega', 'Ώ': 'omega', 'ὠ': 'omega', 'Ὠ': 'omega', 'ὢ': 'omega', 'ὥ': 'omega',
    'ὦ': 'omega', 'ὧ': 'omega', 'ῶ': 'omega',
}

def create_manifest(args):
    glyphs_dir = Path(args.glyphs_dir)
    yolo_model_path = Path(args.yolo_model)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_limit = args.debug_limit

    print(f"Loading authoritative class map from YOLO model: {yolo_model_path}")
    model = YOLO(yolo_model_path)
    id_to_raw_name_map = model.names

    print(f"Scanning '{glyphs_dir}' to create a new, robust manifest...")
    manifest_rows = []
    fieldnames = ["path", "tm_id", "source", "category_id", "base_char_name"]
    
    # Use a more robust regex to find 'cat' followed by digits
    cat_id_pattern = re.compile(r'cat(\d+)')
    
    files_processed = 0
    for glyph_file in tqdm(glyphs_dir.glob("*/*.jpg"), desc="Processing glyphs"):
        
        # --- DEBUGGING LOGIC ---
        if files_processed < debug_limit:
            print(f"\n--- DEBUGGING FILE: {glyph_file.name} ---")
        
        writer_id = glyph_file.parent.name
        
        # Use regex search, which is more robust than splitting
        match = cat_id_pattern.search(glyph_file.stem)
        
        if not match:
            if files_processed < debug_limit:
                print("   [FAIL] Reason: Could not find 'cat<number>' pattern in filename.")
            files_processed += 1
            continue

        try:
            category_id = int(match.group(1))
            if files_processed < debug_limit:
                print(f"   [OK] Parsed Category ID: {category_id}")
        except (ValueError, IndexError):
            if files_processed < debug_limit:
                print("   [FAIL] Reason: Found pattern but could not convert number to integer.")
            files_processed += 1
            continue
            
        raw_char_name = id_to_raw_name_map.get(category_id)
        if not raw_char_name:
            if files_processed < debug_limit:
                print(f"   [FAIL] Reason: Category ID '{category_id}' is not in the YOLO model's class map.")
            files_processed += 1
            continue
        
        if files_processed < debug_limit:
            print(f"   [OK] Found raw name from model: '{raw_char_name}'")

        base_char_name = VARIANT_MAP.get(raw_char_name)
        if not base_char_name:
            if files_processed < debug_limit:
                print(f"   [FAIL] Reason: Raw name '{raw_char_name}' is not in our VARIANT_MAP (likely punctuation or non-Greek).")
            files_processed += 1
            continue
            
        if files_processed < debug_limit:
            print(f"   [OK] Normalized to base name: '{base_char_name}'")
            print("   [SUCCESS] This glyph will be added to the manifest.")

        manifest_rows.append({
            "path": glyph_file.as_posix(),
            "tm_id": writer_id,
            "source": "yolo",
            "category_id": category_id,
            "base_char_name": base_char_name
        })
        files_processed += 1

    print(f"\nFound {len(manifest_rows)} valid glyphs with trustworthy character names.")
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Robust manifest saved to: {output_csv.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a master CSV manifest robustly from glyphs and a YOLO model.")
    parser.add_argument("--glyphs_dir", type=str, required=True, help="Path to the sanitized glyph directory organized by writer.")
    parser.add_argument("--yolo_model", type=str, required=True, help="Path to the trained .pt YOLO model file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output manifest CSV.")
    parser.add_argument("--debug_limit", type=int, default=5, help="Number of files to print detailed debug info for.")
    args = parser.parse_args()
    create_manifest(args)