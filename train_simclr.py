import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import random

import dataset_factory
from simclr import SimCLRModel
from loss import nt_xent_loss

#  Config 
# manifest_path = "glyph_manifest.csv"
manifest_path = "page_independent_splits/train_split.csv"
batch_size = 128
epochs = 100
lr = 3e-4
temperature = 0.7
img_size = 64
save_path = Path("checkpoints/simclr_encoder.pt")
save_path.parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Transforms 
simclr_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#  Data 
dataset = dataset_factory.get_dataset(manifest_path, mode="simclr_pair", transform=simclr_transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#  Model 
model = SimCLRModel(base_model="resnet18", out_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
z1_list = []
#  Training 
for epoch in tqdm(range(1, epochs + 1), desc="Epoch", total=epochs, leave=False):
    model.train()
    total_loss = 0
    z1, z2 = None, None
    loop = tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", leave=False)
    for x1, x2 in loop:
        x1, x2 = x1.to(device), x2.to(device)

        z1 = model(x1)
        z2 = model(x2)

        loss = nt_xent_loss(z1, z2, temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
    with torch.no_grad():
        z1_list.append(z1.norm(dim=1).mean().item())

    if epoch % 10 == 0 or epoch == epochs:
        torch.save(model.encoder.state_dict(), save_path)
        print(f"Encoder checkpoint saved at epoch {epoch}: {save_path}")

for i,v in enumerate(z1_list):
    print(f"[Epoch {i}] z1: {v:.4f}")