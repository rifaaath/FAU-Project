import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import dataset_factory  # Assumes this file exists and is correct
from simclr import SimCLRModel  # Assumes this file exists and is correct
from pathlib import Path


def extract(args):
    checkpoint_path = args.checkpoint_path
    manifest_csv = args.manifest_csv
    output_path = args.output_path
    image_size = 64
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimCLRModel(base_model=args.model_type)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.encoder.load_state_dict(state_dict)
        print("Encoder weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load encoder weights: {e}")
        exit()
    model = model.encoder.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = dataset_factory.get_dataset(path=manifest_csv, mode="writer_cls", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_embeddings, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, labels, sources, paths in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            feats = model(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
            all_embeddings.append(feats)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path,
                        embeddings=np.concatenate(all_embeddings, axis=0),
                        labels=np.array(all_labels),
                        paths=np.array(all_paths))
    print(f"\nSaved {len(all_paths)} embeddings to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract feature embeddings from glyphs using a trained encoder.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet18",
        help="The base model architecture (e.g., resnet18, resnet50)."
    )
    args = parser.parse_args()
    extract(args)