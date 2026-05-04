"""Reconcile 03_flower_rois.csv with the ROI files actually present on disk
after manual curation. Drops rows whose crop_path no longer exists."""

from pathlib import Path
import pandas as pd

CSV = Path("data/csvs/03_flower_rois.csv")
ROOT = Path(".")  # crop_path is stored relative to project root

df = pd.read_csv(CSV)
n_before = len(df)

mask_exists = df["crop_path"].apply(lambda p: (ROOT / p).is_file())
df_clean = df[mask_exists].reset_index(drop=True)

backup = CSV.with_suffix(".csv.bak")
CSV.rename(backup)
df_clean.to_csv(CSV, index=False)

print(f"Rows before: {n_before}")
print(f"Rows after:  {len(df_clean)}")
print(f"Dropped:     {n_before - len(df_clean)}")
print(f"Backup:      {backup}")
