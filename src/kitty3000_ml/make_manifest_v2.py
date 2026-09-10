# MK: The authors' augmentation, used properly. Builds two manifests on top of data/manifest.csv:
# MK:
# MK:   data/manifest_v2_clean.csv   train = originals in train + their _aug1(1) clones.  val / test = originals only, unchanged.
# MK:                                 -> "augmentation done right": no clone of a val/test clip anywhere in train
# MK:   data/manifest_v2_leaky.csv   train = originals in train + ALL clones, incl. those of val/test originals.  val / test unchanged.
# MK:                                 -> the paper's mistake, on our fixed evaluation set. Difference to v2_clean = pure leakage.
# MK:
# MK: Val and test rows are identical to manifest.csv in both files, so every number is measured on the same 448 / 447 clips.
# MK: Clones get the group of their original and split "train". 24 originals have no clone in the zip - they just stay alone.
# MK:
# MK: DATA_DIR for these manifests is data/raw/NAYA_DATA_AUG1X (the V2 zip): it has the originals (byte-identical to
# MK: CatSound_originals) AND the clones in the same folders, so one root works for both.
# MK: run from the repo root:  uv run python src/kitty3000_ml/make_manifest_v2.py

import os
import sys
import pandas as pd

DATA_DIR = "data/raw/NAYA_DATA_AUG1X"
IN_PATH = "data/manifest.csv"
OUT_CLEAN = "data/manifest_v2_clean.csv"
OUT_LEAKY = "data/manifest_v2_leaky.csv"

for out in [OUT_CLEAN, OUT_LEAKY]:
    if os.path.exists(out):
        print(out, "already exists - delete it first if you really want a new one")
        sys.exit(1)

if not os.path.isdir(DATA_DIR):
    print(DATA_DIR, "not found - unzip CatSound_DataSet_V2.zip into data/raw/ first")
    sys.exit(1)

manifest = pd.read_csv(IN_PATH)

# MK: 1. one clone row per original that has a clone on disk. Same label, same group, split = train.
clone_rows = []
missing = 0
for i in range(len(manifest)):
    clone_path = manifest["path"][i].replace(".mp3", "_aug1(1).mp3")
    if os.path.exists(os.path.join(DATA_DIR, clone_path)):
        clone_rows.append({
            "path": clone_path,
            "label": manifest["label"][i],
            "group": manifest["group"][i],
            "split": "train",
            "original_split": manifest["split"][i],   # MK: only used below to decide clean vs leaky
        })
    else:
        missing = missing + 1
clones = pd.DataFrame(clone_rows)
print(len(clones), "clones found,", missing, "originals without a clone")

# MK: 2. clean: only clones whose original is in train. leaky: all clones.
clean_clones = clones[clones["original_split"] == "train"].drop(columns=["original_split"])
leaky_clones = clones.drop(columns=["original_split"])

v2_clean = pd.concat([manifest, clean_clones], ignore_index=True)
v2_leaky = pd.concat([manifest, leaky_clones], ignore_index=True)

# MK: 3. write
v2_clean.to_csv(OUT_CLEAN, index=False)
v2_leaky.to_csv(OUT_LEAKY, index=False)

# MK: 4. print what happened, and check the one thing that must hold: val/test identical to manifest.csv
for name, df in [("manifest.csv", manifest), ("v2_clean", v2_clean), ("v2_leaky", v2_leaky)]:
    counts = df["split"].value_counts()
    print(name.ljust(14), "train", counts["train"], " val", counts["val"], " test", counts["test"])

base_eval = sorted(manifest[manifest["split"] != "train"]["path"])
for name, df in [("v2_clean", v2_clean), ("v2_leaky", v2_leaky)]:
    same = sorted(df[df["split"] != "train"]["path"]) == base_eval
    print(name, "val/test identical to manifest.csv:", same)

leak = (v2_leaky.groupby("group")["split"].nunique() > 1).sum()
print("v2_leaky: groups in more than one split:", leak, "(that is the leak)")
leak = (v2_clean.groupby("group")["split"].nunique() > 1).sum()
print("v2_clean: groups in more than one split:", leak, "(must be 0)")
