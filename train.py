import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import dataset_factory

manifest_path = "glyph_manifest.csv"
batch_size = 64
num_epochs = 20
img_size = 64
lr = 1e-4
num_workers = 4
checkpoint_dir = Path("checkpoints/writer_cls")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


full_dataset = dataset_factory.get_dataset(
    manifest_path,
    mode="writer_cls",
    transform=transform,
    source_filter=None
)


train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_classes = len(full_dataset.tm_to_idx)


#  Model 
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


#  Evaluation 
def evaluate(model, val_loader):
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc="Validating", leave=False)
        for imgs, labels, sources, _ in val_loop:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for pred, label, source in zip(preds, labels, sources):
                correct[source] += (pred == label).item()
                total[source] += 1

    accs = {s: correct[s] / total[s] for s in total}
    for s in accs:
        print(f"[{s}] Accuracy: {accs[s]:.4f}")
    return accs


#  Training 
best_acc = 0.0
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"[Epoch {epoch}]", leave=False)

    for imgs, labels, _, __ in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    val_acc_dict = evaluate(model, val_loader)
    val_acc = sum(val_acc_dict.values()) / len(val_acc_dict)

    print(f"[Epoch {epoch}] Validation Accuracy by source:")
    for src, acc in val_acc_dict.items():
        print(f"  {src:>7}: {acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        ckpt_path = checkpoint_dir / f"best_model_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved best model at epoch {epoch}: {ckpt_path}")

print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
