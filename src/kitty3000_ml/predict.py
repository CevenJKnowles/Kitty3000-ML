from kitty3000_ml.labels import LABELS
from kitty3000_ml.models import dummy, panns, ast, cnn

MODELS = {"dummy": dummy, "panns": panns, "ast": ast, "cnn": cnn}
CAT_THRESHOLD = 0.2


def available_models():
    names = []
    for name in MODELS:
        if MODELS[name].available():
            names.append(name)
    return names


def default_model():
    if panns.available():
        return "panns"
    return "dummy"


def predict(wav_bytes, model_name):
    model = MODELS[model_name]
    result = model.run(wav_bytes)
    scores = result["scores"]
    cat_score = result["cat_score"]

    probs = {}
    for i in range(len(scores)):
        probs[LABELS[i]] = scores[i]

    best_label = ""
    best_score = 0
    for label in probs:
        if probs[label] > best_score:
            best_score = probs[label]
            best_label = label

    is_cat = True
    if cat_score is not None:
        is_cat = cat_score >= CAT_THRESHOLD
    if not is_cat:
        best_label = "Unknown"

    return {"label": best_label, "probs": probs, "cat_score": cat_score, "is_cat": is_cat, "model": model_name}
