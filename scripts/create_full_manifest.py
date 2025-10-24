import csv
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def create_char_normalization_map(json_path):
    """
    Creates a robust, direct map from every category ID to a normalized
    base character name using a comprehensive variant map.
    """
    with open(json_path) as f:
        categories = json.load(f)['categories']

    VARIANT_MAP = {
        # Alpha
        'α': 'alpha', 'Α': 'alpha', 'ᾶ': 'alpha', 'Ά': 'alpha', 'ά': 'alpha', 'ἀ': 'alpha', 'Ἀ': 'alpha',
        'ἂ': 'alpha', 'ἃ': 'alpha', 'ἄ': 'alpha', 'Ἄ': 'alpha', 'ἆ': 'alpha', 'Ἆ': 'alpha', 'ᾱ': 'alpha',
        'Ᾱ': 'alpha', 'ἁ': 'alpha', 'Ἁ': 'alpha', 'a': 'alpha', 'A': 'alpha',
        # Beta
        'β': 'beta', 'Β': 'beta',
        # Gamma
        'γ': 'gamma', 'Γ': 'gamma', 'g': 'gamma',
        # Delta
        'δ': 'delta', 'Δ': 'delta',
        # Epsilon
        'ε': 'epsilon', 'Ε': 'epsilon', 'έ': 'epsilon', 'Έ': 'epsilon', 'ἐ': 'epsilon', 'Ἐ': 'epsilon',
        'ἒ': 'epsilon', 'Ἕ': 'epsilon', 'ἕ': 'epsilon', 'Ἑ': 'epsilon',
        # Zeta
        'ζ': 'zeta', 'Ζ': 'zeta',
        # Eta
        'η': 'eta', 'Η': 'eta', 'ή': 'eta', 'Ή': 'eta', 'ἠ': 'eta', 'Ἠ': 'eta', 'ἢ': 'eta', 'ἣ': 'eta',
        'ἥ': 'eta', 'Ἥ': 'eta', 'ἧ': 'eta', 'Ἧ': 'eta', 'ῆ': 'eta',
        # Theta
        'θ': 'theta', 'Θ': 'theta',
        # Iota
        'ι': 'iota', 'Ι': 'iota', 'ί': 'iota', 'Ί': 'iota', 'ἰ': 'iota', 'Ἰ': 'iota', 'ἒ': 'iota',
        'ἳ': 'iota', 'ἴ': 'iota', 'Ἴ': 'iota', 'ἶ': 'iota', 'Ἶ': 'iota', 'ἱ': 'iota', 'Ἱ': 'iota',
        'ἷ': 'iota', 'ῒ': 'iota', 'ῗ': 'iota', 'ϊ': 'iota', 'ΐ': 'iota', 'Ϊ': 'iota', 'i': 'iota', 'I': 'iota',
        # Kappa
        'κ': 'kappa', 'Κ': 'kappa', 'k': 'kappa',
        # Lambda
        'λ': 'lambda', 'Λ': 'lambda',
        # Mu
        'μ': 'mu', 'Μ': 'mu', 'm': 'mu',
        # Nu
        'ν': 'nu', 'Ν': 'nu', 'n': 'nu',
        # Xi
        'ξ': 'xi', 'Ξ': 'xi',
        # Omicron
        'ο': 'omicron', 'Ο': 'omicron', 'ό': 'omicron', 'Ό': 'omicron', 'ὀ': 'omicron', 'Ὀ': 'omicron',
        'ὃ': 'omicron', 'ὄ': 'omicron', 'Ὄ': 'omicron',
        # Pi
        'π': 'pi', 'Π': 'pi',
        # Rho
        'ρ': 'rho', 'Ρ': 'rho', 'ῥ': 'rho', 'Ῥ': 'rho',
        # Sigma
        'σ': 'sigma', 'ς': 'sigma', 'Σ': 'sigma', 'ϲ': 'sigma', 'Ϲ': 'sigma', 's': 'sigma', 'c': 'sigma',
        # Tau
        'τ': 'tau', 'Τ': 'tau',
        # Upsilon
        'υ': 'upsilon', 'Υ': 'upsilon', 'ύ': 'upsilon', 'Ύ': 'upsilon', 'ϋ': 'upsilon', 'Ϋ': 'upsilon',
        'ὐ': 'upsilon', 'ὒ': 'upsilon', 'ὔ': 'upsilon', 'ὖ': 'upsilon', 'ὑ': 'upsilon', 'Ὑ': 'upsilon',
        'ὓ': 'upsilon', 'ὕ': 'upsilon', 'Ὕ': 'upsilon', 'ὗ': 'upsilon',
        # Phi
        'φ': 'phi', 'Φ': 'phi',
        # Chi
        'χ': 'chi', 'Χ': 'chi',
        # Psi
        'ψ': 'psi', 'Ψ': 'psi',
        # Omega
        'ω': 'omega', 'Ω': 'omega', 'ώ': 'omega', 'Ώ': 'omega', 'ὠ': 'omega', 'Ὠ': 'omega', 'ὢ': 'omega',
        'ὥ': 'omega', 'ὦ': 'omega', 'ὧ': 'omega', 'ῶ': 'omega',
    }

    id_to_base_char = {}
    for cat in categories:
        char_name = VARIANT_MAP.get(cat['name'])
        if char_name:
            id_to_base_char[cat['id']] = char_name

    return id_to_base_char


def create_manifest(args):
    merged_dir = Path(args.merged_dir)
    output_csv = Path(args.output_csv)
    metadata_json = Path(args.metadata_json)

    print("Creating character normalization map...")
    id_to_char_name_map = create_char_normalization_map(metadata_json)

    print(f"Scanning '{merged_dir}' to create a full manifest...")
    manifest_rows = []
    fieldnames = ["path", "tm_id", "source", "category_id", "base_char_name"]

    for tm_folder in tqdm(merged_dir.iterdir(), desc="Processing writers"):
        if not tm_folder.is_dir(): continue
        tm_id = tm_folder.name
        for glyph_file in tm_folder.glob("*.jpg"):
            try:
                parts = glyph_file.stem.split('_')
                cat_id_str = parts[-2].replace('cat', '')  # Adjusted parsing for yolo tag
                category_id = int(cat_id_str)

                base_char_name = id_to_char_name_map.get(category_id)

                if base_char_name:
                    manifest_rows.append({
                        "path": glyph_file.as_posix(),
                        "tm_id": tm_id,
                        "source": "yolo",
                        "category_id": category_id,
                        "base_char_name": base_char_name
                    })
            except (IndexError, ValueError):
                continue

    print(f"\nFound {len(manifest_rows)} valid glyphs with recognized character names.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Full manifest with base_char_name saved to: {output_csv.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a master CSV manifest from an organized glyph directory.")
    parser.add_argument("--merged_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--metadata_json", type=str, required=True)
    args = parser.parse_args()
    create_manifest(args)