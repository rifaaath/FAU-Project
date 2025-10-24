import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupShuffleSplit
import argparse


def create_vlac_descriptors(args):
    embedding_path = Path(args.embeddings)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output_path)
    power_norm = args.power_norm

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and Prepare Data
    print("Loading embeddings and manifest...")
    data = np.load(embedding_path, allow_pickle=True)
    df_main = pd.DataFrame({'path': data["paths"], 'embedding': list(data["embeddings"])})
    df_manifest = pd.read_csv(manifest_path)
    df_main = pd.merge(df_main, df_manifest[['path', 'tm_id', 'base_char_name']], on='path', how='left')

    # Create doc_id for splitting
    def get_doc_id(path):
        parts = Path(path).stem.split('_')
        return f"{parts[0]}_{parts[1]}" if len(parts) > 2 else Path(path).stem

    df_main['doc_id'] = df_main['path'].apply(get_doc_id)
    df_main.dropna(inplace=True)

    # Perform page-indep. split
    print("Performing page-independent split to create a training set...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, _ = next(splitter.split(df_main, groups=df_main['doc_id']))
    df_train = df_main.iloc[train_indices]
    print(f"Using {len(df_train)} glyphs for training the codebook.")

    # Compute the "Codebook" of Mean Character Vectors
    print("Computing mean vector for each character from the training set...")
    char_codebook = {}
    for char_name, group in tqdm(df_train.groupby('base_char_name'), desc="Building codebook"):
        char_codebook[char_name] = np.mean(np.stack(group['embedding'].values), axis=0)

    valid_chars = sorted(list(char_codebook.keys()))
    print(f"Codebook created for {len(valid_chars)} character types.")

    # Create Simple-VLAC Descriptor for Every Page
    print("Creating Simple-VLAC descriptors for all pages...")
    final_page_descriptors = {}

    # Group the entire dataset by document page
    for doc_id, doc_group in tqdm(df_main.groupby('doc_id'), desc="Calculating VLACs"):
        doc_vlac_parts = []
        # Iterate through the official list of characters from our codebook
        for char_name in valid_chars:
            # Get all embeddings for this character on this page
            page_char_embs = doc_group[doc_group['base_char_name'] == char_name]['embedding'].tolist()

            if page_char_embs:
                # Compute the sum of residuals against the single mean vector
                mean_vector = char_codebook[char_name]
                sum_of_residuals = np.sum(np.stack(page_char_embs) - mean_vector, axis=0)
            else:
                # If the character is not on this page, the residual is a zero vector
                sum_of_residuals = np.zeros_like(list(char_codebook.values())[0])

            doc_vlac_parts.append(sum_of_residuals)

        # Concatenate all character parts into one long vector for the page
        full_vlac_vector = np.concatenate(doc_vlac_parts)

        # Power normalization (intra-normalization)
        normalized_vlac = np.sign(full_vlac_vector) * (np.abs(full_vlac_vector) ** power_norm)
        # L2 normalization (global normalization)
        normalized_vlac = normalize(normalized_vlac.reshape(1, -1), norm='l2').flatten()

        final_page_descriptors[doc_id] = normalized_vlac

    # Save the Final Descriptors
    all_doc_ids = list(final_page_descriptors.keys())
    all_descriptors = np.array([final_page_descriptors[did] for did in all_doc_ids])

    # Get writer ID for each doc ID
    doc_to_writer = dict(zip(df_main.doc_id, df_main.tm_id))
    all_writer_ids = [doc_to_writer[did] for did in all_doc_ids]

    np.savez_compressed(
        output_path,
        descriptors=all_descriptors,
        doc_ids=np.array(all_doc_ids),
        writer_ids=np.array(all_writer_ids)  # Save writer IDs for evaluation
    )
    print(f"\nSuccess! Saved {len(all_descriptors)} Simple-VLAC page descriptors to '{output_path}'")
    print(f"   Final descriptor dimension: {all_descriptors.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Simple-VLAC page descriptors.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the FULL embeddings .npz file.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the corresponding FULL manifest CSV.")
    parser.add_argument("--output_path", type=str, default="page_descriptors_simple_vlac.npz",
                        help="Path to save the output descriptors.")
    parser.add_argument("--power_norm", type=float, default=0.5, help="Power normalization factor (alpha).")
    args = parser.parse_args()
    create_vlac_descriptors(args)