import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def filter_npz(args):
    print(f"Loading full embeddings from: {args.full_embeddings_path}")
    full_data = np.load(args.full_embeddings_path, allow_pickle=True)
    path_to_emb = {path: emb for path, emb in zip(full_data['paths'], full_data['embeddings'])}

    print(f"Loading manifest to filter by: {args.manifest_path}")
    df_filter = pd.read_csv(args.manifest_path)
    filter_paths = set(df_filter['path'])

    filtered_embs = []
    filtered_paths = []

    # Iterate through the original order to maintain consistency
    for path in full_data['paths']:
        if path in filter_paths:
            filtered_paths.append(path)
            filtered_embs.append(path_to_emb[path])

    print(f"Found {len(filtered_paths)} matching embeddings.")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_path,
                        embeddings=np.array(filtered_embs),
                        paths=np.array(filtered_paths))

    print(f"Saved filtered embeddings to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_embeddings_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    filter_npz(args)