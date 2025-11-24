# create_manifest_grk.py
import argparse
import csv
import re
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Same variant map as before
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
    crops_dir = Path(args.crops_dir)
    gt_csv = Path(args.gt_csv)
    output_csv = Path(args.output_csv)
    model = YOLO(args.yolo_model)
    id_to_raw_name = model.names

    # Load Ground Truth CSV to get Writer IDs
    print(f"Loading ground truth from {gt_csv}...")
    df_gt = pd.read_csv(gt_csv)
    
    filename_to_writer = {}
    # The CSV column is 'GRK-papyrus' e.g. "Abraamios 1"
    for name in df_gt['GRK-papyrus'].unique():
        clean_filename = name.replace(" ", "_") # "Abraamios_1"
        writer_id = name.split(" ")[0] # "Abraamios"
        filename_to_writer[clean_filename] = writer_id

    print(f"Mapped {len(filename_to_writer)} pages to writers.")

    manifest_rows = []
    cat_pattern = re.compile(r'cat(\d+)')

    # Use glob to find all JPGs
    for crop_path in tqdm(list(crops_dir.glob("*.jpg")), desc="Processing GRK crops"):
        stem = crop_path.stem
        
        # 1. Extract Original Page Name
        if "_glyph_" not in stem: continue
        page_name = stem.split("_glyph_")[0]
        
        writer_id = filename_to_writer.get(page_name)
        if not writer_id: continue

        # 2. Extract Character Class
        match = cat_pattern.search(stem)
        if not match: continue
        cat_id = int(match.group(1))
        
        raw_name = id_to_raw_name.get(cat_id)
        if not raw_name: continue
        
        base_char = VARIANT_MAP.get(raw_name)
        if not base_char: continue

        manifest_rows.append({
            "path": crop_path.as_posix(),
            "tm_id": writer_id, 
            "doc_id": page_name,
            "base_char_name": base_char,
            "source": "yolo"  # <--- FIXED: ADDED THIS KEY
        })

    pd.DataFrame(manifest_rows).to_csv(output_csv, index=False)
    print(f"Created manifest with {len(manifest_rows)} glyphs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops_dir", required=True)
    parser.add_argument("--gt_csv", required=True)
    parser.add_argument("--yolo_model", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    create_manifest(args)