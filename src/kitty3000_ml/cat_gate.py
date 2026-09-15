import librosa
import numpy as np
from panns_inference import AudioTagging, labels as AUDIOSET_LABELS

# MK: PANN's native sample rate. Do not feed it 16 kHz audio.
PANN_SR = 32000

# MK: The five AudioSet classes that are cats, by EXACT name (same as panns.py).
# MK: We do NOT substring-match "cat", because "cat" is also inside "Cattle, bovinae"
# MK: -> a mooing cow would count as a cat. Exact names avoid that.
CAT_LABELS = ["Cat", "Purr", "Meow", "Hiss", "Caterwaul"]
CAT_INDICES = []
for name in CAT_LABELS:
    CAT_INDICES.append(AUDIOSET_LABELS.index(name))

# MK: One shared PANN (CNN14) instance for the whole app, loaded once and lazily.
# MK: panns.py imports get_pann() too, so we never load PANN twice. Sebastian's
# MK: config: checkpoint_path=None (uses ~/panns_data), CPU.
_pann = None


def get_pann():
    global _pann
    if _pann is None:
        _pann = AudioTagging(checkpoint_path=None, device="cpu")
    return _pann


def cat_gate_score(path):
    # MK: load the clip at PANN's sample rate and ask PANN how "cat" it sounds.
    # MK: cat_score is the highest probability among the five cat classes (0..1).
    y, _ = librosa.load(path, sr=PANN_SR, mono=True)
    audio = y[None, :].astype(np.float32)
    clipwise, _ = get_pann().inference(audio)
    cat_score = 0.0
    for i in CAT_INDICES:
        if clipwise[0][i] > cat_score:
            cat_score = float(clipwise[0][i])
    return cat_score
