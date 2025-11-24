import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

MANIFEST = "pipeline_output/5_final_manifest.csv"
OUTPUT = "figures/class_distribution_2.png"

def plot_distribution():
    df = pd.read_csv(MANIFEST)
    
    # Count and take top 25
    counts = df['base_char_name'].value_counts().head(25)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 25 Character Classes in HomerComp (Imbalance)", fontsize=16)
    plt.ylabel("Number of Glyphs")
    plt.xlabel("Character Class")
    
    plt.tight_layout()
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300)
    print(f"Saved distribution plot to {OUTPUT}")

if __name__ == "__main__":
    plot_distribution()