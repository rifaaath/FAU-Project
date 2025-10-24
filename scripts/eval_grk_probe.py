import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse


def train_and_eval_probe(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Training Data 
    train_data = np.load(args.train_embeddings_path, allow_pickle=True)
    df_train_manifest = pd.read_csv(args.train_manifest_path)
    X_train = torch.tensor(train_data['embeddings'], dtype=torch.float32)

    # Create a label map from the writers present in the training set
    train_writers = sorted(df_train_manifest['tm_id'].unique())
    writer_to_label = {writer: i for i, writer in enumerate(train_writers)}
    num_classes = len(writer_to_label)

    y_train = torch.tensor([writer_to_label[w] for w in df_train_manifest['tm_id']], dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    print(f"Training a linear probe on {len(train_writers)} writers using {len(X_train)} glyphs.")

    # Load Test Data 
    test_data = np.load(args.test_embeddings_path, allow_pickle=True)
    df_test_manifest = pd.read_csv(args.test_manifest_path)
    X_test = torch.tensor(test_data['embeddings'], dtype=torch.float32)
    # Use the same label map, but handle writers that might not be in the training set
    y_test_labels = [writer_to_label.get(w, -1) for w in df_test_manifest['tm_id']]
    y_test = torch.tensor(y_test_labels, dtype=torch.long)

    print(f"Evaluating on {len(X_test)} glyphs from the test set.")

    # Model Training 
    model = nn.Linear(X_train.size(1), num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 21):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # Model Evaluation 
    model.eval()
    all_preds = []
    with torch.no_grad():
        logits = model(X_test.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

    # Final Result 
    accuracy = accuracy_score(y_test.numpy(), all_preds) * 100

    print("\n" + "=" * 40)
    print("GRK Papyri 50 - Final Result")
    print("   Protocol: Standard Train/Test Split")
    print(f"   Glyph-level Top-1 Accuracy: {accuracy:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_embeddings_path", required=True)
    parser.add_argument("--train_manifest_path", required=True)
    parser.add_argument("--test_embeddings_path", required=True)
    parser.add_argument("--test_manifest_path", required=True)
    args = parser.parse_args()
    train_and_eval_probe(args)