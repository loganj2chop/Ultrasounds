#!/usr/bin/env python3
import os
import re
import ast
import argparse
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Keep only rows whose 'shape' is a cine loop.")
    ap.add_argument("--csv", required=True, help="Input CSV from the tag scraper (needs 'shape').")
    ap.add_argument("--out-csv", required=True, help="Output CSV with only cine rows.")
    ap.add_argument("--shape-col", default="shape")
    ap.add_argument("--min-frames", type=int, default=2, help="Minimum frames to count as a cine.")
    return ap.parse_args()

def parse_shape(s):
    """Return the shape as a tuple of ints, or None if unparseable.
    Handles tuples/lists, '(108, 899, 1442)', and messy strings like '(300,480.480.3)'."""
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (tuple, list)) and all(isinstance(x, (int, float)) for x in v):
            return tuple(int(x) for x in v)
    except (ValueError, SyntaxError):
        pass
    nums = re.findall(r"\d+", s)          # fallback: pull out every integer
    return tuple(int(x) for x in nums) if nums else None

def n_frames(shape):
    """Number of frames implied by the pixel-array shape (pydicom layout: channels last)."""
    if shape is None:
        return 0
    if len(shape) == 4:                   # (F, H, W, C)
        return shape[0]
    if len(shape) == 3:
        if shape[-1] in (3, 4):           # (H, W, C) -> single color image
            return 1
        return shape[0]                   # (F, H, W) grayscale cine
    return 1                              # (H, W) single image

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.shape_col not in df.columns:
        raise ValueError(f"Column '{args.shape_col}' not found. Columns: {list(df.columns)}")
    print(f"Loaded {len(df)} rows from {args.csv}")

    shapes = df[args.shape_col].apply(parse_shape)
    frames = shapes.apply(n_frames)
    is_cine = frames >= args.min_frames

    n_bad = int(shapes.isna().sum())
    if n_bad:
        print(f"⚠️ {n_bad} rows had an unparseable/empty shape (treated as not cine)")

    out = df[is_cine].copy()                    # same columns as the input
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    dims = shapes.apply(lambda s: len(s) if s else 0)
    print(f"Cine rows: {len(out)} / {len(df)}")
    print(f"  4D (frames x H x W x C): {int((is_cine & (dims == 4)).sum())}")
    print(f"  3D (frames x H x W):     {int((is_cine & (dims == 3)).sum())}")
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()