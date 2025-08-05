# train_and_evaluate_probe_robustness.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Config ---
embedding_path = "embeddings.npz"
batch_size = 64
epochs = 20
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Data ---
print("Loading data...")
data = np.load(embedding_path)
X = torch.tensor(data["embeddings"], dtype=torch.float32)
y_global_labels = torch.tensor(data["labels"], dtype=torch.long)
paths = data["paths"]
sources = np.array([p.split("__")[-1].replace(".jpg", "") for p in paths])

# --- Create a writer map for ALL writers ---
writer_ids = np.array([p.split("/")[-2] for p in paths])
unique_writers = sorted(np.unique(writer_ids))
writer_to_label = {writer: i for i, writer in enumerate(unique_writers)}
y = torch.tensor([writer_to_label[wid] for wid in writer_ids], dtype=torch.long)
num_classes = len(unique_writers)

# --- Standard 80/20 split on the entire dataset ---
dataset = TensorDataset(X, y)
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=y.numpy())

train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# --- Model ---
model = nn.Linear(X.size(1), num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print(f"Training linear probe on {num_classes} writers...")

# --- Training and Evaluation Loop ---
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} Train Loss: {total_loss / len(train_loader):.4f}")

    # --- Evaluate Robustness on Validation Set ---
    model.eval()
    val_preds, val_targets = [], []
    val_sources = []

    # Get the sources for the validation set
    val_sources_all = sources[val_indices]

    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_loader):
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()

            val_preds.extend(preds)
            val_targets.extend(yb.numpy())

    # --- Calculate Per-Source Accuracy ---
    correct_by_source = defaultdict(int)
    total_by_source = defaultdict(int)
    for p, t, s in zip(val_preds, val_targets, val_sources_all):
        if p == t:
            correct_by_source[s] += 1
        total_by_source[s] += 1

    print("--- Validation Accuracy by Source ---")
    overall_correct = sum(correct_by_source.values())
    overall_total = sum(total_by_source.values())
    print(f"  Overall: {overall_correct / overall_total:.4f}")
    for source_name, total_count in sorted(total_by_source.items()):
        acc = correct_by_source[source_name] / total_count
        print(f"  {source_name.capitalize():>7}: {acc:.4f} ({correct_by_source[source_name]}/{total_count})")
    print("-" * 35)