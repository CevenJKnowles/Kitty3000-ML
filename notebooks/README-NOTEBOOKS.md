# Notebooks

One notebook per person and topic, prefixed with initials: `mk-`, `cjk-`, ... Numbered if there is more than one.

## mk-data-review-1.ipynb

Data review of the cleaned dataset. No model, no split yet.

- Counts per class, md5 check for exact duplicates (expected: 0).
- Segment groups: files that are pieces of one recording (`name (1).mp3`, `(2)`, ...) get one group id. They must stay together when we split.
- Clip duration per class. Defence is short, Resting is long - a model can use that as a shortcut => Baseline?

Needs `CatSound_originals.zip` from the team Drive unzipped to `data/raw/CatSound_originals/<class>/*.mp3` (gitignored). Runs in about 10 s.

Team decision 08.09.2026: originals only, no augmented clones.
