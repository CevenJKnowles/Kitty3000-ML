import os
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from panns_inference import AudioTagging, labels
from kitty3000_ml.preprocess import load_clip

#Set path directory
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"

#determine model chosen e.g. cnn + 11s classifier
MODEL_PATH = MODEL_DIR / "panns_cnn14_32000hz_11s_classifier.joblib"

# MK: PANN outputs a probability for each of the 527 kinds of sound in Google's AudioSet list.
# MK: Five of those 527 are cat sounds. We pick them out by name from `labels` (the list of all
# MK: 527 names, part of panns_inference) instead of hardcoding index numbers.
CAT_LABELS = ["Cat", "Purr", "Meow", "Hiss", "Caterwaul"]
CAT_INDICES = []
for name in CAT_LABELS:
    CAT_INDICES.append(labels.index(name))

def available():
    return os.path.exists(MODEL_PATH)

bundle = None
classifier = None
CLASS_NAMES = None
pann = None

if available():
    bundle = joblib.load(MODEL_PATH)

    #extract classifier from loaded model
    classifier = bundle["classifier"]
    CLASS_NAMES = classifier.classes_.tolist()

    #Load PANN model to create features for new audio
    pann = AudioTagging(
        #load pre-trained PANN model for feature creation, None = ~/panns_data like in the notebook
        checkpoint_path=None,
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

    # MK: PANN returns two things: one probability for each of the 527 AudioSet sounds
    # MK: (clipwise, shape 1 x 527) and the embedding. Until now we only used the embedding.
    clipwise, embedding = pann.inference(waveform[None, :])

    #predict probability distribution of classes
    scores = classifier.predict_proba(embedding)[0]

    # MK: cat_score is the highest probability among the five cat sounds (0 to 1).
    # MK: predict.py compares it with CAT_THRESHOLD and returns "Unknown" if it is too low.
    cat_score = 0.0
    for i in CAT_INDICES:
        if clipwise[0][i] > cat_score:
            cat_score = float(clipwise[0][i])

    return {
        "scores": scores.tolist(),
        "cat_score": cat_score,
    }
