import os
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from panns_inference import AudioTagging
from kitty3000_ml.preprocess import load_clip

#Set path directory
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

#determine model chosen e.g. cnn + 11s classifier
MODEL_PATH = MODEL_DIR / "panns_cnn14_32000hz_11s_classifier.joblib"

def available():
    return os.path.exists(MODEL_PATH)

# Load the classifier from models (created by bas-PANN jupyter notebook)
bundle = joblib.load(
    MODEL_DIR / "panns_cnn14_32000hz_7s_classifier.joblib"
)
classifier = bundle["classifier"]
CLASS_NAMES = classifier.classes_.tolist()

#Load PANN model to create features for new audio
pann = AudioTagging(
    #load pre-trained PANN model for feature creation
    checkpoint_path=str(MODEL_DIR / bundle["panns_checkpoint"]),
    device="cpu",)

#new file ->create probability distribution
def run(audio_bytes):
    # Temporarily save the uploaded audio so load_clip can read it
    # not sure if this is the smartest way tbh ->to be checked
    with TemporaryDirectory() as folder:
        path = Path(folder) / "upload.mp3"
        path.write_bytes(audio_bytes)

        waveform = load_clip(
            path,
            sr=bundle["sr"],
            seconds=bundle["seconds"],
        ).astype("float32")

    #create embedding for classifier input
    _, embedding = pann.inference(waveform[None, :])

    #predict probability distribution of classes
    scores = classifier.predict_proba(embedding)[0]

    return {
        "scores": scores.tolist(),
        "cat_score": None,
    }
