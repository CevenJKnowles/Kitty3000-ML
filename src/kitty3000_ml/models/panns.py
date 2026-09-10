import os

MODEL_PATH = ""


def available():
    return False


def run(audio_bytes):
    return {"scores": [], "cat_score": None}
