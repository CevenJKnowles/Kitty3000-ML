import os
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from kitty3000_ml.preprocess import load_clip
from kitty3000_ml.cat_gate import get_pann, cat_gate_score

#Set path directory
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"

#determine model chosen e.g. cnn + 11s classifier
MODEL_PATH = MODEL_DIR / "panns_cnn14_32000hz_11s_classifier.joblib"

# MK: the cat detection (which AudioSet classes count as "cat", and the shared PANN
# MK: instance) now lives in kitty3000_ml/cat_gate.py, so all three models use the same one.

def available():
    return os.path.exists(MODEL_PATH)

bundle = None
classifier = None
CLASS_NAMES = None

if available():
    bundle = joblib.load(MODEL_PATH)

    #extract classifier from loaded model
    classifier = bundle["classifier"]
    CLASS_NAMES = classifier.classes_.tolist()

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

        # MK: same cat gate as AST and CNN, so the cat_score means the same thing
        # MK: for all three models. Call it inside the with, the temp file still exists.
        cat_score = cat_gate_score(path)

    # MK: PANN turns the clip into an embedding; get_pann() is the one shared instance.
    _, embedding = get_pann().inference(waveform[None, :])

    #predict probability distribution of classes
    scores = classifier.predict_proba(embedding)[0]

    return {
        "scores": scores.tolist(),
        "cat_score": cat_score,
    }
