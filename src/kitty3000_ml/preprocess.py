# MK: Shared preprocessing for training AND the API. Whatever the model was trained on, predict.py must do the same thing
# MK: to an upload later, so this file is the only place where audio is loaded, cut and turned into a spectrogram.
# MK: Two functions:
# MK:   load_clip(path, sr, seconds) -> 1D numpy array: mono, resampled to sr, fixed window of `seconds` around the loudest point
# MK:   logmel(y, sr)                -> 2D numpy array: log-mel spectrogram in dB (the "image" for the CNN)
#
# MK: How to use this (CNN, PANNs, AST):
# MK:   manifest = pd.read_csv("data/manifest.csv")
# MK:   train = manifest[manifest["split"] == "train"]
# MK:   val = manifest[manifest["split"] == "val"]          # MK: test: touch it ONCE, before demo day

# MK:   for each row in train / val:
# MK:       path = "data/raw/CatSound_originals/" + row["path"]
# MK:       CNN:    x_temp = load_clip(path, 32000);  X = logmel(x_temp, 32000)
# MK:       PANNs:  X = load_clip(path, 32000)
# MK:       AST:    X = load_clip(path, 16000)
# MK: Don't build spectrograms anywhere else: the API calls exactly these two functions on an upload.

import numpy as np
import librosa

SR = 32000          # MK: team decision 09.09.: 32 kHz for everything. PANNs needs it anyway, AST resamples to 16 kHz itself (load_clip with SR=16000)
SECONDS = 11.0      # MK: team decision 09.09., based on Ceven's duration tests (2/5/7/11 s). Pass a shorter window
                    # MK: explicitly if a model is slow on it, e.g. load_clip(path, 32000, 7.0) for the CNN.


# MK: only used for the baseline CNN (logmel). PANNs computes its own mel inside the model, nothing to match here.
N_FFT = 1024        # MK: Doshi recipe: 1024 FFT window, 512 hop, 64 mel bands. tbd
HOP_LENGTH = 512
N_MELS = 64


def load_clip(path, sr=SR, seconds=SECONDS):
    # MK: load as mono and resample. The raw files are 44.1 kHz stereo mp3, so this is a real step, not a no-op.
    y, sr = librosa.load(path, sr=sr, mono=True)

    window = int(seconds * sr)              # MK: window length in samples

    # MK: find the loudest point: RMS energy per frame, take the frame with the highest energy
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    loudest_frame = int(np.argmax(rms))
    center = loudest_frame * HOP_LENGTH     # MK: frame index -> sample index

    # MK: cut the window around the center, but keep it inside the clip
    start = center - window // 2
    if start < 0:
        start = 0
    if start + window > len(y):
        start = len(y) - window
    if start < 0:
        start = 0                           # MK: clip is shorter than the window
    y = y[start:start + window]

    # MK: shorter clips get zero-padded at the end - at 11 s that is almost every file, fine for PANNs, keep an eye on it for the CNN, shorten the window if to time consuming
    if len(y) < window:
        y = np.pad(y, (0, window - len(y)))

    return y


def logmel(y, sr=SR):
    # MK: mel spectrogram (power) -> dB relative to the loudest bin. Shape: (N_MELS, frames)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
