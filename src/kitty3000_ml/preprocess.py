# MK: Shared preprocessing for training AND the API. Whatever the model was trained on, predict.py must do the same thing
# MK: to an upload later, so this file is the only place where audio is loaded, cut and turned into a spectrogram.
# MK: Two functions:
# MK:   load_clip(path, sr, seconds) -> 1D numpy array: mono, resampled to sampling rate sr, fixed window of `seconds` around the loudest point
# MK:   logmel(y, sr)                -> 2D numpy array: log-mel spectrogram in dB (the "image" data for the CNN)
# MK: PANNs does NOT need logmel (its mel layer is inside the model), it only needs load_clip with sr=32000.

import numpy as np
import librosa

SR = 16000          # MK: default sample rate for the baseline CNN. PANNs wants 32000, pass it explicitly. tbd
SECONDS = 2.0       # MK: default window length, tbd. Ceven's 7 s / 22 kHz files from 08.09. are the same thing as
                    # MK: load_clip(path, sr=22050, seconds=7.0) - we run 2 s vs 7 s on val before we fix the default.


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

    # MK: clips shorter than the window are padded with zeros at the END. This is the one place where silence gets in.
    # MK: With seconds=2 that hits only the ~110 files under 1 s; with seconds=7 it would hit almost every file.
    if len(y) < window:
        y = np.pad(y, (0, window - len(y)))

    return y


def logmel(y, sr=SR):
    # MK: mel spectrogram (power) -> dB relative to the loudest bin. Shape: (N_MELS, frames)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
