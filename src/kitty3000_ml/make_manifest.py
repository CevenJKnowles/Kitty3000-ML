# MK: Builds data/manifest.csv: one row per mp3 in data/raw/CatSound_originals with label, group and split.
# MK: Run ONCE from the repo root:  uv run python src/kitty3000_ml/make_manifest.py

import os
import sys
import re                # MK: regex for group_key
import random
import pandas as pd

DATA_DIR = "data/raw/CatSound_originals"   # MK: one folder per class, mp3 files inside
OUT_PATH = "data/manifest.csv"
SEED = 42                                   # MK: fixed seed, so the split is the same on every machine
VAL_SHARE = 0.15
TEST_SHARE = 0.15                           # MK: train gets the rest (0.70). Team decision 08.09.2026
MAX_GROUP_SIZE_FOR_VAL_TEST = 5             # MK: groups with more files than this always go to train, see below. tbd if we keep this

# MK: 0. safety stop. The manifest is written ONCE and committed. Re-running with a different folder content
# MK:    (or a different seed) gives a different split, and every score anyone measured before is no longer comparable.
# MK:    If you really need a new split, delete data/manifest.csv by hand first and tell the team.
if os.path.exists(OUT_PATH):
    print("STOP:", OUT_PATH, "already exists.")
    print("Re-creating the manifest would change the split and corrupt all results measured so far.")
    print("Delete the file by hand and inform the team if you really want a new split.")
    sys.exit(1)


def group_key(filename):
    # MK: pieces of one recording are named "name (1).mp3", "name (2).mp3", ... -> same cat, same video.
    # MK: They must land in the same split, so they share one group key. Same logic as in mk-data-review-1.ipynb.
    key = filename.replace(".mp3", "")
    key = re.sub(r"\s*\(\d+\)$", "", key)   # MK: remove " (1)", "(2)", ... at the end
    key = key.replace("_opt", "")
    return key.lower()


# MK: 1. collect all files. Everything is sorted, because os.listdir returns a different order on Mac / Linux / Windows.
# MK:    Without sorting the same seed would give a different split on another machine.
paths = []
labels = []
groups = []

classes = sorted(os.listdir(DATA_DIR))
for label in classes:
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        continue                                    # MK: skip README.txt etc.
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".mp3"):
            paths.append(label + "/" + filename)    # MK: relative to DATA_DIR, so the csv works on every machine
            labels.append(label)
            groups.append(label + "/" + group_key(filename))

df = pd.DataFrame()
df["path"] = paths
df["label"] = labels
df["group"] = groups

# MK: 2. split by hand, per class: shuffle the class' groups (fixed seed), then fill test to ~15 %, val to ~15 %, rest train.
# MK:    Whole groups only - pieces of one recording must not land in train AND test, or the model just recognises the cat.
# MK:    Groups bigger than MAX_GROUP_SIZE_FOR_VAL_TEST go to train (the 24-piece Fighting video would swamp its test set).
random.seed(SEED)
split_of_group = {}

for label in classes:
    subset = df[df["label"] == label]
    if len(subset) == 0:
        continue
    group_sizes = subset["group"].value_counts()   # MK: how many files each group has
    group_names = sorted(group_sizes.index)
    random.shuffle(group_names)

    n_files = len(subset)
    test_target = TEST_SHARE * n_files
    val_target = VAL_SHARE * n_files
    n_test = 0
    n_val = 0

    for group in group_names:
        size = group_sizes[group]
        if size > MAX_GROUP_SIZE_FOR_VAL_TEST:
            # MK: the biggest group (Fighting) has 24 pieces of ONE video. In test that would be half of the Fighting test set from one cat, so the score measures one cat, not the class. -> always train.
            split_of_group[group] = "train"
        elif n_test < test_target:
            split_of_group[group] = "test"
            n_test = n_test + size
        elif n_val < val_target:
            split_of_group[group] = "val"
            n_val = n_val + size
        else:
            split_of_group[group] = "train"

splits = []
for group in df["group"]:
    splits.append(split_of_group[group])
df["split"] = splits

# MK: 3. write
os.makedirs("data", exist_ok=True)
df.to_csv(OUT_PATH, index=False)
