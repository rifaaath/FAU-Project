import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Load Data
print("Loading ResNet50 Embeddings...")
data = np.load("pipeline_output/full_embeddings_resnet50_300e.npz", allow_pickle=True)
df = pd.read_csv("pipeline_output/5_final_manifest.csv")

# Filter for TEST set writers only (to show generalization)
# Or use top 20 writers for clarity
top_writers = df['tm_id'].value_counts().head(20).index
indices = df[df['tm_id'].isin(top_writers)].index
embeddings = data['embeddings'][indices]
labels = df.iloc[indices]['tm_id'].values

print(f"Running t-SNE on {len(embeddings)} glyphs from 20 writers...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 10))
sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels, palette="tab20", s=15, legend='full', alpha=0.7)
plt.title("t-SNE of ResNet50 Embeddings (Test Writers)", fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Writer ID")
plt.tight_layout()
plt.savefig("figures/tsne_resnet50.png", dpi=300)
print("Saved figures/tsne_resnet50.png")