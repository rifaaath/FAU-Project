import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import random

class GlyphDataset(Dataset):
    def __init__(self, csv_path, transform=None, label_map=None, source_filter=None):
        self.df = pd.read_csv(csv_path)
        if source_filter:
            self.df = self.df[self.df["source"] == source_filter].reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map or self._create_label_map()
        self.tm_to_idx = self.label_map


    def _create_label_map(self):
        tm_ids = sorted(self.df["tm_id"].unique())
        return {tm: i for i, tm in enumerate(tm_ids)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_map[row["tm_id"]]
        source = row["source"]
        return image, label, source, row["path"]

class SimCLRPairDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Group by TM ID (writer)
        self.tm_to_paths = {}
        for _, row in self.df.iterrows():
            tm = row["tm_id"]
            self.tm_to_paths.setdefault(tm, []).append(row["path"])

        self.tm_ids = list(self.tm_to_paths.keys())

        print(f"[SimCLR] Loaded TM IDs: {len(self.tm_to_paths)}")
        print(f"[SimCLR] Total glyphs: {len(self.df)}")
        for tm, paths in list(self.tm_to_paths.items())[:5]:
            print(f"  TM {tm}: {len(paths)} glyphs")

    def __len__(self):
        return len(self.tm_ids)

    def __getitem__(self, idx):
        tm_id = self.tm_ids[idx]
        glyph_paths = self.tm_to_paths[tm_id]

        # Randomly sample two diff glyphs from this writer
        pair = random.sample(glyph_paths, 2) if len(glyph_paths) >= 2 else [glyph_paths[0], glyph_paths[0]]

        img1 = Image.open(pair[0]).convert("RGB")
        img2 = Image.open(pair[1]).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # print(f"[TM {tm_id}] {pair[0]} vs {pair[1]}")

        return img1, img2  # positive pair

def get_dataset(path, mode="writer_cls", transform=None, **kwargs):
    if mode == "writer_cls":
        return GlyphDataset(path, transform=transform, **kwargs)
    elif mode == "simclr_pair":
        return SimCLRPairDataset(path, transform=transform)
    else:
        raise ValueError(f"Unknown mode {mode}")

