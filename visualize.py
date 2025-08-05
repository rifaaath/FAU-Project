import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import argparse
from tqdm import tqdm

#  Config 
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="embeddings.npz", help="Path to embeddings .npz file")
parser.add_argument("--perplexity", type=int, default=30)
parser.add_argument("--n_samples", type=int, default=30000)
args = parser.parse_args()

#  Load data 
data = np.load(args.input)
X = data["embeddings"]
y = data["labels"]
paths = data["paths"]

# Extract source from filename
sources = np.array([
    "kornia" if "kornia" in str(p).lower() else "opencv" for p in paths
])

print("[Debug] Sample paths and sources:")
for p, s in zip(paths[:5], sources[:5]):
    print(f"  {p} -> {s}")


#  Optional: Sample subset (t-SNE doesn’t scale well) 
if args.n_samples and args.n_samples < len(X):
    indices = np.random.choice(len(X), args.n_samples, replace=False)
    X = X[indices]
    y = y[indices]
    sources = sources[indices]

print(f"[t-SNE] Fitting on {len(X)} samples...")

#  t-SNE 
cluster_params = [[5, 50],
                  [15, 200],
                  [30, 500],
                  [50, 1000]
                  ]
for param in tqdm(cluster_params, total=len(cluster_params)):
    tsne = TSNE(n_components=2, perplexity=param[0], init="pca", random_state=42, learning_rate=param[1])
    X_tsne = tsne.fit_transform(X)

    #  Plot by TM ID 
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(np.unique(y)))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=palette, legend=False, s=10)
    plt.title("p:{}, lr:{}".format(param[0], param[1]))
    plt.tight_layout()
    plt.savefig("tsne_by_tm_pca_{}_{}.png".format(param[0], param[1]))
    plt.close()

    #  Plot by Source (Kornia/OpenCV) 
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=sources, palette="deep", s=10)
    plt.title("p:{}, lr:{}".format(param[0], param[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig("tsne_by_source_pca_{}_{}.png".format(param[0], param[1]))

    print("Saved: tsne_by_tm.png and tsne_by_source.png for pca")
