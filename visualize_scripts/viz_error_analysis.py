import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def generate_error_analysis(args):
    manifest_path = Path(args.manifest)
    target_writer = args.writer_id
    target_char = args.char_name
    output_file = Path(args.output)
    
    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    # Filter for specific writer and character
    df_subset = df[
        (df['tm_id'] == target_writer) & 
        (df['base_char_name'] == target_char)
    ].copy()
    
    if len(df_subset) < 10:
        print(f"Error: Not enough samples for {target_writer} - {target_char} (Found {len(df_subset)})")
        return

    print(f"Found {len(df_subset)} samples for {target_writer} ('{target_char}').")
    print("Loading images and clustering by visual similarity...")

    # Load images and flatten for clustering
    images = []
    valid_paths = []
    
    for path in df_subset['path']:
        try:
            # Read in grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # Resize for consistency in clustering
            img_resized = cv2.resize(img, (64, 64))
            
            images.append(img_resized.flatten())
            valid_paths.append(path)
        except Exception:
            continue
            
    X = np.array(images)
    
    # --- CLUSTERING ---
    # We want to show "Multi-modal distribution" (two different styles)
    # So we force k=2 clusters to find the two distinct modes
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Get indices for Cluster 0 and Cluster 1
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]
    
    # Select random samples from each cluster (up to 9 for a 3x3 grid)
    n_samples = 9
    samples_0 = np.random.choice(idx_0, min(len(idx_0), n_samples), replace=False)
    samples_1 = np.random.choice(idx_1, min(len(idx_1), n_samples), replace=False)
    
    print(f"Cluster A size: {len(idx_0)} | Cluster B size: {len(idx_1)}")

    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Intra-Writer Variability Analysis: {target_writer} ('{target_char}')", fontsize=16, y=0.95)
    
    # Subplot layout: Left block (Style A), Right block (Style B)
    # We manually place grids
    
    def plot_grid(sample_indices, start_col_idx, title):
        # 3x3 grid logic
        for i, idx in enumerate(sample_indices):
            if i >= 9: break
            path = valid_paths[idx]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate subplot position
            row = i // 3
            col = i % 3
            
            # Left grid occupies subplot slots 1,2,3, 5,6,7... etc is hard to map
            # Using subplot2grid is easier
            ax = plt.subplot2grid((3, 7), (row, col + start_col_idx))
            ax.imshow(img)
            ax.axis('off')
            
            if i == 1: # Title above middle column
                ax.set_title(title, fontsize=12, pad=10)

    # Plot Cluster A (Early/Style 1) at column offset 0
    plot_grid(samples_0, 0, "Visual Mode A (e.g. 'Early')")
    
    # Plot separator line in middle column (col index 3)
    ax_sep = plt.subplot2grid((3, 7), (0, 3), rowspan=3)
    ax_sep.plot([0.5, 0.5], [0, 1], color='gray', linestyle='--', linewidth=2)
    ax_sep.set_xlim(0, 1)
    ax_sep.set_ylim(0, 1)
    ax_sep.axis('off')

    # Plot Cluster B (Late/Style 2) at column offset 4
    plot_grid(samples_1, 4, "Visual Mode B (e.g. 'Late')")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize intra-writer variation.")
    parser.add_argument("--manifest", required=True, help="Path to full manifest csv")
    parser.add_argument("--writer_id", type=str, default="TM_60220", help="Writer ID to analyze")
    parser.add_argument("--char_name", type=str, default="alpha", help="Character to visualize (alpha, epsilon...)")
    parser.add_argument("--output", type=str, default="figures/error_analysis_tm60220.png")
    args = parser.parse_args()
    generate_error_analysis(args)