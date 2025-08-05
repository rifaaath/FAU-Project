# dataset_factory_supervised.py
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random
from collections import defaultdict


class SupervisedContrastiveDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # --- Group glyphs by (writer_id, category_id) ---
        self.data_map = defaultdict(list)
        print("Grouping glyphs by writer and character class...")
        for _, row in self.df.iterrows():
            # Create a unique key for each class (e.g., 'TM_60214_cat_8')
            class_key = f"{row['tm_id']}_cat_{row['category_id']}"
            self.data_map[class_key].append(row['path'])

        # Filter out classes with only one sample, as they can't form positive pairs
        self.class_keys = [k for k, v in self.data_map.items() if len(v) >= 2]

        # Create a flat list of all glyphs for easy negative sampling
        self.all_glyphs = self.df[['path', 'tm_id']].to_records(index=False)

        print(f"Found {len(self.class_keys)} writer/character classes with 2+ samples.")

    def __len__(self):
        return len(self.all_glyphs)

    def __getitem__(self, idx):
        # Anchor: The glyph at the current index
        anchor_path, anchor_writer = self.all_glyphs[idx]

        # --- Find a Positive Sample ---
        # 1. Get the category of the anchor
        anchor_row = self.df.loc[self.df['path'] == anchor_path].iloc[0]
        anchor_cat = anchor_row['category_id']
        anchor_class_key = f"{anchor_writer}_cat_{anchor_cat}"

        # 2. Sample another glyph from the same writer AND same category
        positive_list = self.data_map[anchor_class_key]
        positive_path = random.choice(positive_list)
        # Ensure we don't pick the exact same file as the anchor
        while positive_path == anchor_path and len(positive_list) > 1:
            positive_path = random.choice(positive_list)

        # --- Find a Negative Sample ---
        # Sample any random glyph until we find one from a different writer
        negative_path, negative_writer = self.all_glyphs[random.randint(0, len(self.all_glyphs) - 1)]
        while negative_writer == anchor_writer:
            negative_path, negative_writer = self.all_glyphs[random.randint(0, len(self.all_glyphs) - 1)]

        # --- Load and Transform Images ---
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# You can add a get_dataset function if you want to keep the factory pattern