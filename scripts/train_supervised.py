import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

from dataset_factory_supervised import SupervisedContrastiveDataset
from simclr import SimCLRModel  # Your existing SimCLR model is fine


train_manifest_path = "pipeline_output/6_final_splits/train_final.csv"
batch_size = 128
epochs = 300
lr = 3e-4
margin = 0.2  # Triplet loss margin
img_size = 64
save_path = Path("checkpoints/supcon_encoder_resnet50_300e_dropout.pt")
save_path.parent.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Data 
print("Loading Supervised Contrastive Dataset...")
dataset = SupervisedContrastiveDataset(train_manifest_path, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Model 
model = SimCLRModel(base_model="resnet50", out_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Use the Triplet Margin Loss
criterion = nn.TripletMarginLoss(margin=margin)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

# Training Loop 
for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Training Epoch {epoch}/{epochs}", leave=False)
    for anchor_img, positive_img, negative_img in loop:
        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)

        # Get embeddings
        anchor_emb = model(anchor_img)
        positive_emb = model(positive_img)
        negative_emb = model(negative_img)

        # Calculate loss
        loss = criterion(anchor_emb, positive_emb, negative_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    scheduler.step()
    avg_loss = total_loss / len(loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}| LR: {current_lr:.6f}")

    if epoch % 10 == 0 or epoch == epochs:
        torch.save(model.encoder.state_dict(), save_path)
        print(f"Supervised Contrastive encoder saved: {save_path}")