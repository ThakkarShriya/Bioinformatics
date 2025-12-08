import os
from PIL import Image

def resize_all(folder_path, size=(128, 128)):
    for filename in os.listdir(folder_path):
        if filename.endswith('.gif'):
            path = os.path.join(folder_path, filename)
            img = Image.open(path).convert("L")
            img = img.resize(size)
            img.save(path)

# Resize both healthy and MCI images
resize_all("dataset/healthy")
resize_all("dataset/mci")

print("✅ All images resized to 128x128.")
