from PIL import Image
import os

broken = []
total = 0
for root, dirs, files in os.walk("papytwin/HomerCompTraining/images/"):
    for file in files:
        if file.lower().endswith(".jpg"):
            total += 1
            path = os.path.join(root, file)
            try:
                with Image.open(path) as im:
                    im.verify()
            except:
                broken.append(path)

print(f"Broken files: {len(broken)}/{total}")
