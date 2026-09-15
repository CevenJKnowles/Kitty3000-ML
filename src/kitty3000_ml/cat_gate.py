import librosa
import numpy as np
import torch
from panns_inference import AudioTagging, labels as AUDIOSET_LABELS

PANN_SR = 32000  # PANN's native/expected sample rate - do not feed it 16kHz audio

CAT_LABEL_KEYWORDS = ["cat", "meow", "purr", "hiss", "caterwaul", "growling"]

_device = "cuda" if torch.cuda.is_available() else "cpu"
_pann_gate = None
_cat_label_indices = None


def _load():
    global _pann_gate, _cat_label_indices
    if _pann_gate is None:
        _pann_gate = AudioTagging(checkpoint_path=None, device=_device)
        _cat_label_indices = [
            i for i, label in enumerate(AUDIOSET_LABELS)
            if any(kw in label.lower() for kw in CAT_LABEL_KEYWORDS)
        ]
        print(f"[cat_gate] matched {len(_cat_label_indices)} AudioSet labels: "
              f"{[AUDIOSET_LABELS[i] for i in _cat_label_indices]}")


def cat_gate_score(path):
    _load()
    y, _ = librosa.load(path, sr=PANN_SR, mono=True)
    audio = y[None, :].astype(np.float32)
    audioset_scores, _ = _pann_gate.inference(audio)
    return float(np.max(audioset_scores[0, _cat_label_indices]))
