import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def analyze_distinctiveness(args):
    """
    Calculates a 'distinctiveness score' for each writer based on how much their
    average form of a specific character deviates from the global average.
    """
    embedding_path = Path(args.embeddings)
    manifest_path = Path(args.manifest)
    target_char = args.character
    output_csv_path = Path(args.output)
    min_samples = args.min_samples

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and Merge Data 
    print("Loading embeddings and manifest data...")
    try:
        data = np.load(embedding_path, allow_pickle=True)
        # Use a dictionary for fast lookup
        path_to_embedding = {path: emb for path, emb in zip(data["paths"], data["embeddings"])}

        df_manifest = pd.read_csv(manifest_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Details: {e}")
        return

    # Filter manifest for paths that actually have an embedding
    df_manifest = df_manifest[df_manifest['path'].isin(path_to_embedding.keys())]

    # Add the embedding vector directly to the DataFrame
    df_manifest['embedding'] = df_manifest['path'].map(path_to_embedding)

    print(f"Loaded and merged data for {len(df_manifest)} glyphs.")

    # Step 2: Isolate Target Character and Calculate Global Mean 
    print(f"Analyzing distinctiveness for character: '{target_char}'")
    df_char = df_manifest[df_manifest['base_char_name'] == target_char].copy()

    if df_char.empty:
        print(f"Error: No data found for character '{target_char}' in the manifest.")
        return

    all_char_embs = np.stack(df_char['embedding'].values)
    global_mean_char = np.mean(all_char_embs, axis=0)

    print(f"Calculated global mean from {len(df_char)} samples of '{target_char}'.")

    # Step 3: Calculate each writer's average and distance to global mean 
    results = []

    # Group by writer ID and iterate
    for writer_id, group in tqdm(df_char.groupby('tm_id'), desc="Processing writers"):
        # Check if the writer has enough samples for a stable average
        if len(group) < min_samples:
            continue

        writer_embs = np.stack(group['embedding'].values)
        writer_mean_char = np.mean(writer_embs, axis=0)

        # Calculate cosine distance (1 - similarity)
        distance = 1 - cosine_similarity(
            writer_mean_char.reshape(1, -1),
            global_mean_char.reshape(1, -1)
        )[0, 0]

        results.append({
            "Writer_ID": writer_id,
            "Distinctiveness_Score": distance,
            "Num_Samples": len(group)
        })

    # Step 4: Save to CSV for analysis 
    if not results:
        print(f"\nCould not generate analysis. No writers had at least {min_samples} samples of '{target_char}'.")
        return

    df_results = pd.DataFrame(results)
    # Sort by the most distinctive (highest score) writers first
    df_results = df_results.sort_values(by="Distinctiveness_Score", ascending=False)

    df_results.to_csv(output_csv_path, index=False)

    print(f"\nAnalysis complete. Results saved to '{output_csv_path.resolve()}'")
    print(f"\nTop 10 Most Distinctive Writers for '{target_char.capitalize()}' ")
    print(df_results.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze writer distinctiveness for a specific character.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to the embeddings .npz file (e.g., test_embeddings_sanitized.npz)."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the corresponding manifest CSV for the embeddings (e.g., test_split.csv)."
    )
    parser.add_argument(
        "--character",
        type=str,
        default="epsilon",
        help="The base character name to analyze (e.g., 'alpha', 'epsilon')."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="character_distinctiveness.csv",
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="Minimum number of character samples a writer must have to be included in the analysis."
    )

    args = parser.parse_args()
    analyze_distinctiveness(args)