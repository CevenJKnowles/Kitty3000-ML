import os
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa
import torch
import torch.nn as nn

from kitty3000_ml.labels import LABELS
from kitty3000_ml.preprocess import SR, logmel
from kitty3000_ml.cat_gate import cat_gate_score

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_PATH = MODEL_DIR / "cnn_baseline" / "cnn_baseline_best.pt"


class CNNBaseline(nn.Module):
    def __init__(self, n_classes=len(LABELS), dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _load_clip_natural(path, sr=SR):
    # Each clip's own full, unmodified waveform - no fixed window, no crop, no
    # pad, no loop. Matches what CatSoundSpectrogramDatasetNatural evaluated
    # on in cjk-CNN-model-testing.ipynb, not the fixed-window load_clip().
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def _normalize_mel(mel):
    return (mel - mel.mean()) / (mel.std() + 1e-6)


def available():
    return os.path.exists(MODEL_PATH)


model = None

if available():
    model = CNNBaseline()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()


def run(audio_bytes):
    # Temporarily save the uploaded audio so librosa can read it
    with TemporaryDirectory() as folder:
        path = Path(folder) / "upload.mp3"
        path.write_bytes(audio_bytes)
        y = _load_clip_natural(path)
        gate_score = cat_gate_score(path)  # same temp file, gate loads its own 32kHz copy internally

    mel = logmel(y)
    mel = _normalize_mel(mel)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    return {"scores": probs.tolist(), "cat_score": gate_score}  # is_cat is computed centrally in predict.py
