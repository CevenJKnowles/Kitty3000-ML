from kitty3000_ml.labels import LABELS # MK: labels.py with dummy labels from the original data, tbd if we keep these
from kitty3000_ml.model import run # MK: model.py comes from the ML-team, which eventually will be replaced with the actual model code and returns the scores for each label

def predict(wav_bytes):
    # MK: This is the main function that will be called by the API, it will call the model and return the label with the highest score and the scores for each label
    scores = run(wav_bytes)

    probs = {} # MK: dictionary to hold the scores for each label
    for i in range(len(LABELS)): # MK: iterate over the labels and scores and add them to the dictionary
        label = LABELS[i]
        score = scores[i]
        probs[label] = score

    best_label = ""
    best_score = 0
    for label in probs: # MK: iterate over the dictionary and find the label with the highest score
        if probs[label] > best_score:
            best_score = probs[label]
            best_label = label

    result = {"label": best_label, "probs": probs} # MK: return the label with the highest score and the scores for each label
    return result
