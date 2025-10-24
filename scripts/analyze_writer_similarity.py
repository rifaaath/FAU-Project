import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_similarity(args):
    """
    Calculates a similarity matrix between all writer prototypes and visualizes
    it as a clustered heatmap.
    """
    embedding_path = Path(args.embeddings)
    manifest_path = Path(args.manifest)
    output_plot_path = Path(args.output)
    min_samples = args.min_samples

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and Merge Data 
    print("Loading embeddings and manifest data...")
    try:
        data = np.load(embedding_path, allow_pickle=True)
        path_to_embedding = {path: emb for path, emb in zip(data["paths"], data["embeddings"])}
        df_manifest = pd.read_csv(manifest_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Details: {e}")
        return

    df_manifest['embedding'] = df_manifest['path'].map(path_to_embedding)
    df_manifest.dropna(subset=['embedding'], inplace=True)
    print(f"Loaded and merged data for {len(df_manifest)} glyphs.")

    # Step 2: Calculate Writer Prototypes 
    print(f"Calculating writer prototypes (average hand)...")
    writer_prototypes = {}

    # Group by writer and calculate the mean embedding
    for writer_id, group in tqdm(df_manifest.groupby('tm_id'), desc="Processing writers"):
        if len(group) < min_samples:
            continue
        writer_prototypes[writer_id] = np.mean(np.stack(group['embedding'].values), axis=0)

    if not writer_prototypes:
        print(f"Error: No writers found with at least {min_samples} glyphs.")
        return

    writer_ids = list(writer_prototypes.keys())
    prototype_matrix = np.array(list(writer_prototypes.values()))
    print(f"Calculated prototypes for {len(writer_ids)} writers.")

    # Step 3: Compute Pairwise Similarity Matrix 
    print("Computing similarity matrix between all writers...")
    similarity_matrix = cosine_similarity(prototype_matrix)

    # For better visualization, let's convert it to a DataFrame
    df_similarity = pd.DataFrame(similarity_matrix, index=writer_ids, columns=writer_ids)

    # Step 4: Visualize as a Clustered Heatmap 
    print("Generating clustered heatmap...")
    # A larger figure size is needed for many writers
    # The size should be proportional to the number of writers
    fig_size = max(12, len(writer_ids) * 0.2)

    # Using clustermap automatically performs hierarchical clustering and reorders the matrix
    g = sns.clustermap(
        df_similarity,
        method='average',  # Clustering algorithm
        metric='cosine',  # Distance metric for clustering (1-similarity)
        cmap='viridis',  # Color map for the heatmap
        figsize=(fig_size, fig_size),
        linewidths=.5,
        cbar_pos=(0.02, 0.8, 0.05, 0.18)  # Position the color bar
    )

    g.fig.suptitle('Hierarchically Clustered Heatmap of Writer Similarity', fontsize=20, y=0.98)
    # Rotate the x-axis labels for better readability
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nWriter similarity heatmap saved to '{output_plot_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and visualize the similarity network of all writers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the FULL embeddings .npz file.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the corresponding FULL manifest CSV.")
    parser.add_argument("--output", type=str, default="writer_similarity_heatmap.png",
                        help="Path to save the output heatmap.")
    parser.add_argument("--min_samples", type=int, default=20,
                        help="Minimum number of glyphs a writer must have to be included.")

    args = parser.parse_args()
    analyze_similarity(args)