import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

MANIFEST = "pipeline_output/5_final_manifest.csv"
OUTPUT = "figures/dataset_samples.png"
CHARS_TO_SHOW = ['alpha', 'epsilon', 'eta', 'iota', 'kappa', 'nu', 'omicron', 'sigma', 'tau', 'upsilon']

def visualize_tags():
    df = pd.read_csv(MANIFEST)
    
    plt.figure(figsize=(20, 3))
    plt.suptitle("HomerComp Dataset: Representative Glyphs", fontsize=20, y=1.05)
    
    found_count = 0
    for i, char in enumerate(CHARS_TO_SHOW):
        # Robust filtering
        subset = df[df['base_char_name'] == char]
        
        if len(subset) == 0:
            continue
            
        # Get a high quality sample (not random, try to get one from the middle)
        sample_path = subset.iloc[len(subset)//2]['path']
        
        if not Path(sample_path).exists(): continue

        img = Image.open(sample_path).convert("RGB")
        
        plt.subplot(1, len(CHARS_TO_SHOW), found_count+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(char.capitalize(), fontsize=14)
        found_count += 1
        
    plt.tight_layout()
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300)
    print(f"Saved dataset samples to {OUTPUT}")

if __name__ == "__main__":
    visualize_tags()