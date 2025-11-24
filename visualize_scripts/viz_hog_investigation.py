import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import cv2
import pandas as pd
from pathlib import Path
import random

MANIFEST = "pipeline_output/5_final_manifest.csv"
OUTPUT = "figures/hog_visualization.png"

def visualize_hog():
    df = pd.read_csv(MANIFEST)
    
    # Pick a random sample
    sample_path = df.sample(1).iloc[0]['path']
    
    # Load image
    img = cv2.imread(sample_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize to 64x64 (same as your pipeline)
    gray_resized = cv2.resize(gray, (64, 64))
    
    # Compute HOG
    fd, hog_image = hog(gray_resized, 
                        orientations=9, 
                        pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), 
                        visualize=True)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(gray_resized, cmap=plt.cm.gray)
    ax1.set_title('Input Image (64x64)')

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.inferno)
    ax2.set_title('HOG Features (Gradients)')
    
    plt.suptitle("Visualization of HOG Features\n(Notice how it captures background texture gradients)", fontsize=14)
    
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=300)
    print(f"Saved HOG viz to {OUTPUT}")

if __name__ == "__main__":
    visualize_hog()