import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_map_for_subset(embeddings, writers):
    """A helper function to run mAP evaluation on a given subset of data."""
    if len(embeddings) < 2:
        return 0.0

    average_precisions = []
    for i in range(len(embeddings)):
        query_emb = embeddings[i]
        true_writer_id = writers[i]

        temp_gallery_embs = np.delete(embeddings, i, axis=0)
        temp_gallery_writers = np.delete(writers, i)

        num_relevant_docs = np.sum(temp_gallery_writers == true_writer_id)
        if num_relevant_docs == 0:
            continue

        similarities = cosine_similarity(query_emb.reshape(1, -1), temp_gallery_embs)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        ranked_writers = temp_gallery_writers[sorted_indices]

        hits = 0
        ap = 0.0
        for k, predicted_writer in enumerate(ranked_writers):
            if predicted_writer == true_writer_id:
                hits += 1
                ap += hits / (k + 1)

        if num_relevant_docs > 0:
            average_precisions.append(ap / num_relevant_docs)

    return np.mean(average_precisions) * 100 if average_precisions else 0.0


def analyze_character_difficulty(args, overall_map_score):
    """
    Calculates the writer identification mAP for each character type individually
    to determine which characters are most stylistically informative.
    """
    embedding_path = Path(args.embeddings)
    manifest_path = Path(args.manifest)
    output_csv_path = Path(args.output_csv)
    output_plot_path = Path(args.output_plot)
    min_samples = args.min_samples

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and Merge Data 
    print("Loading test set embeddings and manifest...")
    try:
        data = np.load(embedding_path, allow_pickle=True)
        df_manifest = pd.read_csv(manifest_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Details: {e}")
        return

    # Create a DataFrame with all necessary info
    df_data = pd.DataFrame({
        'path': data['paths'],
        'embedding': list(data['embeddings'])
    })
    # Merge with manifest to get writer and character names
    df_full = pd.merge(df_data, df_manifest[['path', 'tm_id', 'base_char_name']], on='path', how='left')
    df_full.dropna(inplace=True)
    print(f"Loaded and merged data for {len(df_full)} glyphs.")

    # Step 2: Iterate and Evaluate for Each Character 
    char_results = []
    # Get a list of all characters present in the test set
    characters_to_test = df_full['base_char_name'].unique()

    for char_name in tqdm(characters_to_test, desc="Evaluating characters"):
        df_char = df_full[df_full['base_char_name'] == char_name]

        # Ensure we have enough data to perform a meaningful evaluation
        if len(df_char) < min_samples:
            continue

        # Get the embeddings and writer IDs for this character only
        char_embeddings = np.stack(df_char['embedding'].values)
        char_writers = df_char['tm_id'].values

        # Run the mAP evaluation on this subset
        char_map = evaluate_map_for_subset(char_embeddings, char_writers)

        char_results.append({
            "Character": char_name.capitalize(),
            "mAP_Score": char_map,
            "Num_Glyphs": len(df_char)
        })

    # Step 3: Save and Plot Results 
    if not char_results:
        print(f"Error: No character types met the minimum sample threshold of {min_samples}.")
        return

    df_results = pd.DataFrame(char_results).sort_values(by="mAP_Score", ascending=False)
    df_results.to_csv(output_csv_path, index=False)
    print(f"\nCharacter difficulty analysis complete. Results saved to '{output_csv_path}'")

    # Create the Plot 
    plt.style.use('seaborn-talk')
    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("viridis", n_colors=len(df_results))

    ax = sns.barplot(
        x="mAP_Score",
        y="Character",
        data=df_results,
        palette=palette,
        orient='h'
    )

    plt.title("Writer Identification Performance by Character (Stylistic Richness)", fontsize=18, pad=20)
    plt.xlabel("mAP Score (%)", fontsize=14)
    plt.ylabel("Character", fontsize=14)
    ax.invert_yaxis()

    plt.xlim(0, 100)
    plt.axvline(x=overall_map_score, color='r', linestyle='--', label=f'Overall mAP ({overall_map_score:.2f}%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"Character difficulty plot saved to '{output_plot_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze writer ID performance for each character type.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the TEST embeddings .npz file.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the corresponding TEST manifest CSV.")
    parser.add_argument("--output_csv", type=str, default="character_difficulty.csv",
                        help="Path to save the output CSV.")
    parser.add_argument("--output_plot", type=str, default="character_difficulty_plot.png",
                        help="Path to save the output plot.")
    parser.add_argument("--min_samples", type=int, default=50,
                        help="Minimum number of glyphs a character must have in the test set to be evaluated.")

    # This is a dummy argument for the main mAP score for plotting the average line.
    # We'll calculate it properly in the script if not provided.
    parser.add_argument("--overall_map", type=float, default=87.31,
                        help="Overall mAP score to draw as a reference line.")

    args = parser.parse_args()

    # A bit of a hack to pass the overall mAP to the plotting function
    # In a real application, you might calculate it first or pass it in.
    # For now, let's just add it to the final DataFrame for the plotting function.
    df_full = pd.read_csv(args.manifest)  # Re-reading for simplicity
    df_full['mAP_Score'] = args.overall_map

    analyze_character_difficulty(args, args.overall_map)