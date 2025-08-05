import os
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import dataset_factory

from simclr import SimCLRModel

#  Config 
checkpoint_path = "checkpoints/supcon_encoder.pt"
# manifest_csv = "glyph_manifest.csv"
# output_path = "embeddings.npz"
manifest_csv = "page_independent_splits/test_split.csv"
output_path = "page_indep_embeddings.npz"
image_size = 64
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Load model 
model = SimCLRModel(base_model="resnet18")
state_dict = torch.load(checkpoint_path)

try:
    model.encoder.load_state_dict(state_dict)
    print("Encoder weights loaded")
except RuntimeError as e:
    print(f"Failed to load encoder weights: {e}")
    exit()

model = model.encoder.to(device).eval()  # only encoder used for embeddings

#  Transform and Dataset 
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = dataset_factory.get_dataset(
    path=manifest_csv,
    mode="writer_cls",
    transform=transform,
    # source_filter="kornia"  # or "opencv", or None
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#  Extract and Save Embeddings 
all_embeddings = []
all_labels = []
all_paths = []

with torch.no_grad():
    for imgs, labels, sources, paths in tqdm(loader, desc="Extracting embeddings"):
        imgs = imgs.to(device)
        feats = model(imgs)
        feats = feats.squeeze(-1).squeeze(-1).cpu().numpy()

        all_embeddings.append(feats)
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)

# Save
np.savez_compressed(output_path,
                    embeddings=np.concatenate(all_embeddings, axis=0),
                    labels=np.array(all_labels),
                    paths=np.array(all_paths))

print(f"Saved {len(all_paths)} embeddings to {output_path}")
