# train_probe_on_known_writers.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path # <--- ✅ 1. IMPORT Path

# --- Config ---
embedding_path = "embeddings.npz"
batch_size = 64
epochs = 20
lr = 1e-3
# --- ✅ 2. CORRECTED DEFINITION ---
# Wrap the string in Path() to create a Path object
model_save_path = Path("checkpoints/writer_probe.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ✅ 3. THE REST OF THE CODE NOW WORKS ---
model_save_path.parent.mkdir(parents=True, exist_ok=True)

# --- Load Embeddings & Paths ---
print("Loading data...")
data = np.load(embedding_path)
X = torch.tensor(data["embeddings"], dtype=torch.float32)
paths = data["paths"]

# --- Create Writer-Disjoint Split ---
writer_ids = np.array([p.split("/")[-2] for p in paths])
unique_writers = np.unique(writer_ids)
np.random.seed(42) # for reproducibility
np.random.shuffle(unique_writers)

train_writer_ids = unique_writers[:int(0.8 * len(unique_writers))]

# --- Create a label map for TRAINING writers ONLY ---
train_writer_to_label = {writer_id: i for i, writer_id in enumerate(train_writer_ids)}
num_train_classes = len(train_writer_to_label)
print(f"Training a probe on {num_train_classes} known writers.")

# --- Create training dataset with re-mapped labels ---
train_mask = np.isin(writer_ids, train_writer_ids)
X_train = X[train_mask]
y_train_original_ids = writer_ids[train_mask]
y_train_mapped = torch.tensor([train_writer_to_label[wid] for wid in y_train_original_ids], dtype=torch.long)

train_ds = TensorDataset(X_train, y_train_mapped)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# --- Model: Output dimension MUST match the number of training classes ---
model = nn.Linear(X.size(1), num_train_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Training Loop ---
print("Starting training...")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"[Epoch {epoch}] Train Loss: {total_loss / len(loop):.4f}")

# --- Save the trained model ---
torch.save(model.state_dict(), model_save_path)
print(f"Trained linear probe saved to {model_save_path}")