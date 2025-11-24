import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load Manifest
df = pd.read_csv("pipeline_output/5_final_manifest.csv")

# Count classes
class_counts = df['base_char_name'].value_counts().head(25) # Top 25 classes

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title("Distribution of Top 25 Character Classes in HomerComp", fontsize=15)
plt.xlabel("Character Class", fontsize=12)
plt.ylabel("Number of Glyphs", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Save
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/tag_distribution.png", dpi=300)
print("Saved figures/tag_distribution.png")