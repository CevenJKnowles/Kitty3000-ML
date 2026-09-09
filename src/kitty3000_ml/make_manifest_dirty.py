# MK: "dirty" split for the bad-apple experiment (meeting 09.09.): same files, same labels, same 70/15/15 as
# MK: data/manifest.csv, but the split ignores the (n) segment groups - plain random stratified, like the 2018 paper.
# MK: Run a model once with manifest.csv and once with manifest_dirty.csv, the gap between the two numbers is the leakage.
# MK: run from the repo root:  uv run python src/kitty3000_ml/make_manifest_dirty.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
IN_PATH = "data/manifest.csv"
OUT_PATH = "data/manifest_dirty.csv"

if os.path.exists(OUT_PATH):
    print(OUT_PATH, "already exists - delete it first if you really want a new one")
    sys.exit(1)

manifest = pd.read_csv(IN_PATH)

# MK: 70 / 30 first, then the 30 into 15 / 15. stratify on label, groups ignored on purpose
train, rest = train_test_split(manifest, test_size=0.30, stratify=manifest["label"], random_state=SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)

manifest["split"] = "train"
manifest.loc[val.index, "split"] = "val"
manifest.loc[test.index, "split"] = "test"

manifest.to_csv(OUT_PATH, index=False)
print("wrote", OUT_PATH)
print(manifest["split"].value_counts())
print((manifest.groupby("group")["split"].nunique() > 1).sum(), "groups in more than one split (that is the point here)")
