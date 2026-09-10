DUMMY_SCORES = [0.05, 0.03, 0.02, 0.62, 0.04, 0.02, 0.06, 0.03, 0.08, 0.05]


def available():
    return True


def run(audio_bytes):
    return {"scores": DUMMY_SCORES, "cat_score": None}
