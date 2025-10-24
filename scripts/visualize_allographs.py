import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_character_styles(args):
    """
    Performs k-means clustering on the embeddings of a single character to discover
    allographs and visualizes them using t-SNE.
    """
    embedding_path = Path(args.embeddings)
    manifest_path = Path(args.manifest)
    target_char = args.character
    n_clusters = args.n_clusters
    tsne_samples = args.tsne_samples
    output_plot_path = Path(args.output)

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and Filter Data 
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

    print(f"Filtering for character: '{target_char}'...")
    df_char = df_manifest[df_manifest['base_char_name'] == target_char]

    if df_char.empty:
        print(f"Error: No data found for character '{target_char}' in the manifest.")
        return

    char_embs = np.stack(df_char['embedding'].values)
    print(f"Found {len(char_embs)} samples of '{target_char}'.")

    # Step 2: Discover Allographs with K-Means Clustering 
    if len(char_embs) < n_clusters:
        print(f"Error: Not enough samples ({len(char_embs)}) to train k-means with k={n_clusters}.")
        return

    print(f"Running k-means with {n_clusters} clusters to find allographs...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10, batch_size=256)
    cluster_assignments = kmeans.fit_predict(char_embs)
    cluster_centers = kmeans.cluster_centers_

    # Step 3: Visualize the "Style Space" with t-SNE 
    # Use a subset for t-SNE if the dataset is very large, as it's computationally expensive
    if len(char_embs) > tsne_samples:
        print(f"Running t-SNE on a random subset of {tsne_samples} samples for speed...")
        indices = np.random.choice(len(char_embs), tsne_samples, replace=False)
        embs_for_tsne = char_embs[indices]
        labels_for_tsne = cluster_assignments[indices]
    else:
        print("Running t-SNE on all samples...")
        embs_for_tsne = char_embs
        labels_for_tsne = cluster_assignments

    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, random_state=42, init='pca', learning_rate=200.0)
    embs_2d = tsne.fit_transform(embs_for_tsne)

    # Find the 2D position of the cluster centers 
    # We do this by projecting the 3D centers into the 2D space learned by t-SNE
    centers_2d = []
    for i in range(n_clusters):
        center_3d = cluster_centers[i]
        # Calculate distance from the 3D center to all 3D embeddings used in t-SNE
        distances = np.linalg.norm(embs_for_tsne - center_3d, axis=1)
        closest_point_idx = np.argmin(distances)
        # The 2D position of the center is the 2D position of its closest neighbor
        centers_2d.append(embs_2d[closest_point_idx])
    centers_2d = np.array(centers_2d)

    # Step 4: Create the Plot 
    print("Generating plot...")
    plt.style.use('seaborn-talk')
    plt.figure(figsize=(16, 12))

    palette = sns.color_palette("hsv", n_clusters)

    sns.scatterplot(
        x=embs_2d[:, 0], y=embs_2d[:, 1],
        hue=labels_for_tsne,
        palette=palette,
        legend=False,
        s=15, alpha=0.7
    )

    plt.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        marker='X', s=200, c='black',
        edgecolor='white', linewidth=1.5,
        label='k-means Cluster Centers (Allographs)'
    )

    plt.title(f"t-SNE Visualization of Discovered Allographs for '{target_char.capitalize()}' (k={n_clusters})",
              fontsize=20, pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(output_plot_path, dpi=300)
    print(f"\nAllograph visualization saved to '{output_plot_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover and visualize character allographs using k-means and t-SNE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the FULL embeddings .npz file.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the corresponding FULL manifest CSV.")
    parser.add_argument("--character", type=str, default="epsilon", help="The base character name to analyze.")
    parser.add_argument("--n_clusters", type=int, default=16,
                        help="The number of allographs (k-means clusters) to discover.")
    parser.add_argument("--tsne_samples", type=int, default=5000,
                        help="Max number of samples to use for t-SNE visualization for speed.")
    parser.add_argument("--output", type=str, default="allograph_visualization.png",
                        help="Path to save the output plot.")

    args = parser.parse_args()
    visualize_character_styles(args)