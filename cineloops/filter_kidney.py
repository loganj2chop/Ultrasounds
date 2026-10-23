#!/usr/bin/env python3
import os
import re
import glob
import argparse
import pandas as pd

PATTERNS = {
    'kidney':     r'\b(kidney|kid|rk|lk)\b',
    'bladder':    r'\b(bladder|bla)\b',
    'right':      r'\b(right|rk|rt)\b',
    'left':       r'\b(left|lt|lk)\b',
    'sagittal':   r'\b(sagittal|sag|sg)\b',
    'transverse': r'\b(transverse|trns|trans|trv)\b',
    'liver':      r'\b(liver|lvr)\b',
}

def parse_args():
    ap = argparse.ArgumentParser(description="Filter OCR output to kidney-only images.")
    ap.add_argument("--csv", required=True, nargs="+",
                    help="OCR CSV(s); globs allowed, e.g. '/path/choponlyocr*.csv'.")
    ap.add_argument("--out-csv", required=True, help="Output CSV with kidney-only rows.")
    ap.add_argument("--text-col", default="dicomtext")
    return ap.parse_args()

def main():
    args = parse_args()

    # ---- load (expand globs, concat chunks) ----
    paths = sorted({p for pat in args.csv for p in glob.glob(pat)})
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.csv}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(paths)} file(s)")
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Columns: {list(df.columns)}")

    # Ensure text column is string and NaNs won't break .str.count
    txt = df[args.text_col].fillna('').astype(str)

    for col, rgx in PATTERNS.items():
        df[col] = txt.str.count(rgx, flags=re.IGNORECASE)

    filtered = df[
        (df["kidney"] == 1) &
        (df["liver"] == 0) &
        (df["bladder"] == 0) &
        ((df["left"] == 1) ^ (df["right"] == 1))   # exactly one of left/right is 1
    ].copy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    filtered.to_csv(args.out_csv, index=False)

    print(f"Kidney-only rows: {len(filtered)} / {len(df)}")
    print(f"  right: {int((filtered['right'] == 1).sum())}   left: {int((filtered['left'] == 1).sum())}")
    print(f"  sagittal: {int((filtered['sagittal'] > 0).sum())}   transverse: {int((filtered['transverse'] > 0).sum())}")
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()