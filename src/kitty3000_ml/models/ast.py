import os
import tempfile
import joblib
import torch
import librosa
from transformers import ASTFeatureExtractor, ASTModel
from kitty3000_ml.models.preprocess import load_clip

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "ast_logreg.joblib")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_bundle = None
_feature_extractor = None
_ast_model = None


def _load():
    global _bundle, _feature_extractor, _ast_model
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
        _feature_extractor = ASTFeatureExtractor.from_pretrained(_bundle["checkpoint"])
        _ast_model = ASTModel.from_pretrained(_bundle["checkpoint"]).to(_device)
        _ast_model.eval()


def available():
    try:
        _load()
        return True
    except Exception:
        return False


def _extract_embedding(waveform, sr):
    inputs = _feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _ast_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


def run(audio_bytes):
    _load()

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        waveform = load_clip(tmp.name, sr=_bundle["sr"], seconds=_bundle["seconds"])

    embedding = _extract_embedding(waveform, _bundle["sr"])
    probs = _bundle["model"].predict_proba(embedding[None, :])[0]

    scores = sorted(
        [{"label": c, "score": float(p)} for c, p in zip(_bundle["classes"], probs)],
        key=lambda x: x["score"],
        reverse=True,
    )

    return {"scores": scores, "cat_score": scores[0]}
