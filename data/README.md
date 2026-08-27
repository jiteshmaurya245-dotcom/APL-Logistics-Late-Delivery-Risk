# data/

Place `APL_Logistics.csv` in this folder before running the app or the
training script. The raw dataset is not committed to the repo (see
`.gitignore`) since it's large and not something we want tracked in git
history.

If you run `scripts/train_model.py`, it will also write `model.pkl`,
`scaler.pkl`, `feature_cols.json`, and `metrics.json` here — these are
git-ignored too, since they're derived artifacts you can regenerate at
any time from the CSV.
