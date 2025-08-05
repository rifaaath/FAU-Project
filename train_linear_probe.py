#  train_linear_probe.py 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from matplotlib import pyplot as plt
from collections import defaultdict
from tqdm import tqdm

#  Load embeddings 
data = np.load("embeddings.npz")
X = torch.tensor(data["embeddings"], dtype=torch.float32)
y = torch.tensor(data["labels"], dtype=torch.long)
sources = data["paths"]  # or "sources" if you saved it that way

#  Dataset & Split 
dataset = TensorDataset(X, y)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

#  Model 
num_classes = y.max().item() + 1
model = nn.Linear(X.size(1), num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#  Training Loop 
for epoch in range(1, 21):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch}] Loss: {total_loss / len(train_loader):.4f}")

    #  Evaluation 
    model.eval()
    preds, targets, srcs = [], [], []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_loader):
            xb = xb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            targets.extend(yb.tolist())

            batch_indices = val_ds.indices[i * 64 : (i + 1) * 64]
            srcs.extend([sources[j] for j in batch_indices])

    acc = accuracy_score(targets, preds)
    print(f"[Epoch {epoch}] Validation Accuracy: {acc:.4f}")

    # cm = confusion_matrix(targets, preds, normalize="true")
    #
    # cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # cm_disp.plot(cmap="Blues", xticks_rotation=90)
    # plt.title("Confusion Matrix - Epoch {}".format(epoch))
    # plt.tight_layout()
    # plt.savefig("cm_e_{}.png".format(epoch))
    # plt.close()

    #  Per-source Accuracy 
    correct = defaultdict(int)
    total = defaultdict(int)
    for p, t, s in zip(preds, targets, srcs):
        source_tag = s.split("__")[-1].replace(".jpg", "")  # extract 'opencv' or 'kornia'
        correct[source_tag] += int(p == t)
        total[source_tag] += 1

    print("Validation Accuracy by source:")
    for s in sorted(correct.keys()):
        print(f"  {s:>7}: {correct[s] / total[s]:.4f}")

    print(classification_report(targets, preds, zero_division=0))
