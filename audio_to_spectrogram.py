import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

AUDIO_DIR = r"D:\CNN_Project\CNN_Project\nsynth_dataset\audio"
OUT_DIR   = r"D:\CNN_Project\CNN_Project\spectrograms"

os.makedirs(OUT_DIR, exist_ok=True)

for instrument in os.listdir(AUDIO_DIR):
    inst_path = os.path.join(AUDIO_DIR, instrument)
    if not os.path.isdir(inst_path):
        continue

    out_inst_path = os.path.join(OUT_DIR, instrument)
    os.makedirs(out_inst_path, exist_ok=True)

    for file in tqdm(os.listdir(inst_path), desc=instrument):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(inst_path, file)

        try:
            # Load audio (mono = single channel)
            y, sr = librosa.load(file_path, sr=22050, mono=True)

            # Mel spectrogram
            S = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128,
                fmax=8000
            )

            # Convert to dB (log scale)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Save as grayscale image
            plt.figure(figsize=(3, 3))
            librosa.display.specshow(S_dB, sr=sr)
            plt.axis("off")

            out_file = os.path.join(
                out_inst_path,
                file.replace(".wav", ".png")
            )

            plt.savefig(out_file, bbox_inches="tight", pad_inches=0)
            plt.close()

        except Exception as e:
            print(f"Error with {file}: {e}")
