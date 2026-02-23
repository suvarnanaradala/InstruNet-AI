import os
from PIL import Image
from tqdm import tqdm

# Input folder (grayscale spectrograms)
INPUT_DIR = r"D:\CNN_Project\CNN_Project\spectrograms"
# Output folder (resized images)
OUTPUT_DIR = r"D:\CNN_Project\CNN_Project\spectrograms_resized"

TARGET_SIZE = (128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for inst in os.listdir(INPUT_DIR):
    inst_path = os.path.join(INPUT_DIR, inst)
    if not os.path.isdir(inst_path):
        continue

    out_inst_path = os.path.join(OUTPUT_DIR, inst)
    os.makedirs(out_inst_path, exist_ok=True)

    for file in tqdm(os.listdir(inst_path), desc=inst):
        if not file.endswith(".png"):
            continue

        img_path = os.path.join(inst_path, file)
        img = Image.open(img_path).convert("L")  # ensure grayscale
        img_resized = img.resize(TARGET_SIZE)
        img_resized.save(os.path.join(out_inst_path, file))
