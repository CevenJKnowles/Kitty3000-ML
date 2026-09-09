# MK: "bad apple" manifest - the 2018 paper setup, on purpose. ALL files from the V2 zip: the originals plus their
# MK: _aug1(1) copies (5922 mp3, the 8 exact duplicates included), random stratified 70/15/15, groups ignored.
# MK: Most clones land in another split than their original, so train and test share the same recordings.
# MK: That is where the paper's 91 % comes from. Run a model on this and on manifest.csv, the gap is the story.
# MK:
# MK: Same columns as manifest.csv. A model needs two lines to run on it:
# MK:     DATA_DIR = "data/raw/NAYA_DATA_AUG1X"
# MK:     MANIFEST = "data/manifest_badapple.csv"
# MK: Needs the V2 zip from the Drive (Data/CatSound_DataSet_V2.zip) unzipped into data/raw/:
# MK:     unzip CatSound_DataSet_V2.zip -d data/raw/        -> data/raw/NAYA_DATA_AUG1X/<class>/*.mp3
# MK: run from the repo root:  uv run python src/kitty3000_ml/make_manifest_badapple.py

import os
import sys
import re
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data/raw/NAYA_DATA_AUG1X"
OUT_PATH = "data/manifest_badapple.csv"
SEED = 42

if os.path.exists(OUT_PATH):
    print(OUT_PATH, "already exists - delete it first if you really want a new one")
    sys.exit(1)

if not os.path.isdir(DATA_DIR):
    print(DATA_DIR, "not found - unzip CatSound_DataSet_V2.zip into data/raw/ first")
    sys.exit(1)


def group_key(filename):
    # MK: same as make_manifest.py, plus the _aug1(1) suffix: the clone belongs to the group of its original
    key = filename.replace(".mp3", "")
    key = key.replace("_aug1(1)", "")
    key = re.sub(r"\s*\(\d+\)$", "", key)
    key = key.replace("_opt", "")
    return key.lower()


# MK: 1. collect all files, sorted (same reason as in make_manifest.py: same order on every machine)
paths = []
labels = []
groups = []
is_aug = []

classes = sorted(os.listdir(DATA_DIR))
for label in classes:
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        continue                                    # MK: skip the csv / txt / hdf5 that ship in the zip
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".mp3"):
            paths.append(label + "/" + filename)
            labels.append(label)
            groups.append(label + "/" + group_key(filename))
            is_aug.append("_aug1" in filename)

manifest = pd.DataFrame()
manifest["path"] = paths
manifest["label"] = labels
manifest["group"] = groups

# MK: 2. random split, 70 / 30 first, then the 30 into 15 / 15. stratify on label, groups ignored on purpose
train, rest = train_test_split(manifest, test_size=0.30, stratify=manifest["label"], random_state=SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)

manifest["split"] = "train"
manifest.loc[val.index, "split"] = "val"
manifest.loc[test.index, "split"] = "test"

# MK: 3. write
manifest.to_csv(OUT_PATH, index=False)
print("wrote", OUT_PATH)
print(manifest["split"].value_counts())
print(sum(is_aug), "of", len(manifest), "files are _aug1(1) clones")

# MK: 4. how bad is the apple: groups (original + clone + segments) that are spread over more than one split
groups_split = manifest.groupby("group")["split"].nunique()
print((groups_split > 1).sum(), "of", len(groups_split), "groups are in more than one split (that is the point here)")
