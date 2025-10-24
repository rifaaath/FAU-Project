import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def analyze_consistency_and_distance(args):
    embedding_path = Path(args.embeddings)
    manifest_path = Path(args.manifest)
    char_a_name = args.char_a
    char_b_name = args.char_b
    output_csv_path = Path(args.output)
    min_samples = args.min_samples

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and Merge Data 
    print("Loading embeddings and manifest data...")
    try:
        data = np.load(embedding_path, allow_pickle=True)
        path_to_embedding = {path: emb for path, emb in zip(data["paths"], data["embeddings"])}
        df_manifest = pd.read_csv(manifest_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Details: {e}")
        return

    df_manifest = df_manifest[df_manifest['path'].isin(path_to_embedding.keys())]
    df_manifest['embedding'] = df_manifest['path'].map(path_to_embedding)
    print(f"Loaded and merged data for {len(df_manifest)} glyphs.")

    # Step 2: Calculate Consistency and Distance for Each Writer 
    results = []
    all_writers = df_manifest['tm_id'].unique()

    print(f"\nAnalyzing consistency for {char_a_name.capitalize()} vs. distance from {char_b_name.capitalize()}...")

    for writer_id in tqdm(all_writers, desc="Analyzing writers"):
        df_writer = df_manifest[df_manifest['tm_id'] == writer_id]

        # Get embeddings for Char A and Char B for this writer
        embs_a = df_writer[df_writer['base_char_name'] == char_a_name]['embedding'].tolist()
        embs_b = df_writer[df_writer['base_char_name'] == char_b_name]['embedding'].tolist()

        # We need enough samples of each to do a meaningful comparison
        if len(embs_a) < min_samples or len(embs_b) < min_samples:
            continue

        embs_a = np.stack(embs_a)
        embs_b = np.stack(embs_b)

        # Metric 1: Intra-Letter Cohesion (Consistency) 
        # How similar are this writer's 'char_a' glyphs to their own average 'char_a'?
        mean_char_a = np.mean(embs_a, axis=0).reshape(1, -1)
        cohesion_char_a = np.mean(cosine_similarity(embs_a, mean_char_a))

        # Metric 2: Inter-Letter Distance 
        # How different is this writer's average 'char_a' from their average 'char_b'?
        mean_char_b = np.mean(embs_b, axis=0).reshape(1, -1)
        distance_a_b = 1 - cosine_similarity(mean_char_a, mean_char_b)[0, 0]

        results.append({
            "Writer_ID": writer_id,
            f"{char_a_name.capitalize()}_Cohesion": cohesion_char_a,
            f"{char_a_name.capitalize()}_{char_b_name.capitalize()}_Distance": distance_a_b,
            f"Num_{char_a_name.capitalize()}": len(embs_a),
            f"Num_{char_b_name.capitalize()}": len(embs_b)
        })

    # Step 3: Save to CSV for analysis 
    if not results:
        print(
            f"\nCould not generate analysis. No writers had at least {min_samples} samples of both '{char_a_name}' and '{char_b_name}'.")
        return

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=f"{char_a_name.capitalize()}_{char_b_name.capitalize()}_Distance",
                                        ascending=False)

    df_results.to_csv(output_csv_path, index=False)

    print(f"\nAnalysis complete. Results saved to '{output_csv_path.resolve()}'")
    print(f"\nTop 5 Writers by {char_a_name.capitalize()}-{char_b_name.capitalize()} Distance ")
    print(df_results.head(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze writer consistency for one character vs. distance to another.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the embeddings .npz file.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the corresponding manifest CSV.")
    parser.add_argument("--char_a", type=str, default="epsilon", help="The character to measure consistency for.")
    parser.add_argument("--char_b", type=str, default="alpha", help="The character to measure distance from.")
    parser.add_argument("--output", type=str, default="consistency_vs_distance.csv",
                        help="Path to save the output CSV file.")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="Minimum number of samples for both char_a and char_b.")

    args = parser.parse_args()
    analyze_consistency_and_distance(args)