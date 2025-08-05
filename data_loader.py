from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = GlyphDataset("glyph_manifest.csv", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check a batch
images, labels, sources = next(iter(loader))
print(images.shape, labels[:5], sources[:5])
