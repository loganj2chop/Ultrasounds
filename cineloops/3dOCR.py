#!/usr/bin/env python3
import os
import ast
import math
import gc
import inspect
import time
import argparse
import numpy as np
import pandas as pd
import pydicom
import gcsfs
import torch
import torch.serialization

# ============================
# ARGS
# ============================
def parse_args():
    ap = argparse.ArgumentParser(description="EasyOCR on the first frame of each DICOM in a CSV.")
    ap.add_argument("--csv", required=True, help="Input CSV (needs 'shape', 'directory', 'file_name').")
    ap.add_argument("--out-dir", required=True, help="Directory for output chunk CSVs.")
    ap.add_argument("--model-dir", default="/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/.EasyOCR",
                    help="EasyOCR model storage directory.")
    ap.add_argument("--out-prefix", default="choponlyocr", help="Prefix for output chunk files.")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--langs", nargs="+", default=["en"])
    ap.add_argument("--no-gpu", action="store_true", help="Run EasyOCR on CPU.")
    return ap.parse_args()

# ============================
# SHAPE PARSING
# ============================
def shape_ok(v):
    if not isinstance(v, (tuple, list)):
        return False
    if len(v) in (2, 3):
        return True
    if len(v) == 4 and v[-1] == 3:
        return True
    return False

# ============================
# TORCH LOAD PATCH
# ============================
def patch_torch_load():
    torch.load = torch.serialization.load
    if "weights_only" not in inspect.signature(torch.load).parameters:
        _orig = torch.load
        def _compat(*args, **kwargs):
            kwargs.pop("weights_only", None)
            return _orig(*args, **kwargs)
        torch.load = _compat

# ============================
# GCS
# ============================
_gcsfs = None
def get_gcsfs():
    global _gcsfs
    if _gcsfs is None:
        _gcsfs = gcsfs.GCSFileSystem(token="google_default")
    return _gcsfs

def gs_to_key(gs_path):
    return gs_path.replace("gs://", "", 1)

# ============================
# SAFE DICOM READERS
# ============================
def dcmread_local_with_retries(path, retries=3, delay=0.5):
    for i in range(retries):
        try:
            return pydicom.dcmread(path, force=True)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(delay)

def dcmread_any(path):
    if isinstance(path, str) and path.startswith("gs://"):
        fs = get_gcsfs()
        with fs.open(gs_to_key(path), "rb") as f:
            return pydicom.dcmread(f, force=True)
    else:
        return dcmread_local_with_retries(path)

# ============================
# IMAGE HELPERS
# ============================
def apply_rescale(arr, ds):
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr.astype(np.float32) * slope + intercept

def window_uint8(arr, ds):
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])

    x = arr.astype(np.float32)

    if wc is not None and ww not in (None, 0):
        lo, hi = wc - ww / 2, wc + ww / 2
        x = np.clip(x, lo, hi)
        x = (x - lo) / max(hi - lo, 1e-6) * 255
    else:
        mn, mx = np.nanmin(x), np.nanmax(x)
        x = (x - mn) / max(mx - mn, 1e-6) * 255

    return x.astype(np.uint8)

# ============================
# FIRST FRAME EXTRACTOR
# ============================
def get_first_frame(ds):
    arr = ds.pixel_array
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    samples = int(getattr(ds, "SamplesPerPixel", 1))

    if arr.ndim == 2:
        img = window_uint8(apply_rescale(arr, ds), ds)
        if "MONOCHROME1" in photometric:
            img = 255 - img
        return img

    if arr.ndim == 3 and samples == 3 and arr.shape[-1] == 3:
        return arr.astype(np.uint8)

    if arr.ndim == 3 and samples == 1:
        img = window_uint8(apply_rescale(arr[0], ds), ds)
        if "MONOCHROME1" in photometric:
            img = 255 - img
        return img

    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr[0].astype(np.uint8)

    return None

# ============================
# OCR
# ============================
def ocr_from_dicom(path, reader):
    try:
        ds = dcmread_any(path)
        img = get_first_frame(ds)

        if img is None:
            print("⚠️ Unsupported shape:", ds.pixel_array.shape)
            return ""

        print(f"OCR on {path} | shape={img.shape}")
        result = reader.readtext(img, detail=0, paragraph=True)
        return "\n".join(result).strip() if result else ""

    except Exception as e:
        print("❌ OCR error:", path, "|", e)
        return ""

# ============================
# MAIN
# ============================
def main():
    args = parse_args()
    use_gpu = not args.no_gpu

    print("Input CSV:", args.csv)
    print("Output dir:", args.out_dir)
    print("GPU Enabled:", use_gpu)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load input CSV ----
    df = pd.read_csv(args.csv)
    print("Loaded rows:", len(df))

    df["shape"] = df["shape"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else s
    )
    df = df[df["shape"].apply(shape_ok)].copy()
    print("Rows after shape filter:", len(df))

    df["pathtodicom"] = df["directory"] + "/" + df["file_name"]
    df1 = df.reset_index(drop=True)

    # ---- EasyOCR (local models only) ----
    patch_torch_load()
    import easyocr
    reader = easyocr.Reader(
        args.langs,
        gpu=use_gpu,
        model_storage_directory=args.model_dir
    )

    # ---- process in chunks ----
    n = len(df1)
    n_chunks = math.ceil(n / args.chunk_size)
    print(f"Total rows: {n} | Chunks: {n_chunks}")

    for chunk_idx in range(n_chunks):
        start = chunk_idx * args.chunk_size
        stop = min(start + args.chunk_size, n)
        out_path = os.path.join(args.out_dir, f"{args.out_prefix}{chunk_idx+1}.csv")

        print(f"\n[Chunk {chunk_idx+1}/{n_chunks}] rows {start}-{stop-1}")
        chunk = df1.iloc[start:stop].copy()

        texts = []
        for i, row in chunk.iterrows():
            texts.append(ocr_from_dicom(row["pathtodicom"], reader))

        chunk["dicomtext"] = texts
        chunk.to_csv(out_path, index=False)
        print("Saved:", out_path)

        del chunk, texts
        gc.collect()

    print("DONE.")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
import os
import ast
import math
import gc
import inspect
import time
import argparse
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.serialization

# ============================
# ARGS
# ============================
def parse_args():
    ap = argparse.ArgumentParser(description="EasyOCR on the first frame of each DICOM in a CSV.")
    ap.add_argument("--csv", required=True, help="Input CSV (needs 'shape', 'directory', 'file_name').")
    ap.add_argument("--out-dir", required=True, help="Directory for output chunk CSVs.")
    ap.add_argument("--model-dir", default="/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/.EasyOCR",
                    help="EasyOCR model storage directory.")
    ap.add_argument("--out-prefix", default="choponlyocr", help="Prefix for output chunk files.")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--langs", nargs="+", default=["en"])
    ap.add_argument("--no-gpu", action="store_true", help="Run EasyOCR on CPU.")
    return ap.parse_args()

# ============================
# SHAPE PARSING
# ============================
def shape_ok(v):
    if not isinstance(v, (tuple, list)):
        return False
    if len(v) in (2, 3):
        return True
    if len(v) == 4 and v[-1] == 3:
        return True
    return False

# ============================
# TORCH LOAD PATCH
# ============================
def patch_torch_load():
    torch.load = torch.serialization.load
    if "weights_only" not in inspect.signature(torch.load).parameters:
        _orig = torch.load
        def _compat(*args, **kwargs):
            kwargs.pop("weights_only", None)
            return _orig(*args, **kwargs)
        torch.load = _compat

# ============================
# SAFE DICOM READER (local paths only)
# ============================
def dcmread_any(path, retries=3, delay=0.5):
    for i in range(retries):
        try:
            return pydicom.dcmread(path, force=True)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(delay)

# ============================
# IMAGE HELPERS
# ============================
def apply_rescale(arr, ds):
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr.astype(np.float32) * slope + intercept

def window_uint8(arr, ds):
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])

    x = arr.astype(np.float32)

    if wc is not None and ww not in (None, 0):
        lo, hi = wc - ww / 2, wc + ww / 2
        x = np.clip(x, lo, hi)
        x = (x - lo) / max(hi - lo, 1e-6) * 255
    else:
        mn, mx = np.nanmin(x), np.nanmax(x)
        x = (x - mn) / max(mx - mn, 1e-6) * 255

    return x.astype(np.uint8)

# ============================
# FIRST FRAME EXTRACTOR
# ============================
def get_first_frame(ds):
    arr = ds.pixel_array
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    samples = int(getattr(ds, "SamplesPerPixel", 1))

    if arr.ndim == 2:
        img = window_uint8(apply_rescale(arr, ds), ds)
        if "MONOCHROME1" in photometric:
            img = 255 - img
        return img

    if arr.ndim == 3 and samples == 3 and arr.shape[-1] == 3:
        return arr.astype(np.uint8)

    if arr.ndim == 3 and samples == 1:
        img = window_uint8(apply_rescale(arr[0], ds), ds)
        if "MONOCHROME1" in photometric:
            img = 255 - img
        return img

    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr[0].astype(np.uint8)

    return None

# ============================
# OCR
# ============================
def ocr_from_dicom(path, reader):
    try:
        ds = dcmread_any(path)
        img = get_first_frame(ds)

        if img is None:
            print("⚠️ Unsupported shape:", ds.pixel_array.shape)
            return ""

        print(f"OCR on {path} | shape={img.shape}")
        result = reader.readtext(img, detail=0, paragraph=True)
        return "\n".join(result).strip() if result else ""

    except Exception as e:
        print("❌ OCR error:", path, "|", e)
        return ""

# ============================
# MAIN
# ============================
def main():
    args = parse_args()
    use_gpu = not args.no_gpu

    print("Input CSV:", args.csv)
    print("Output dir:", args.out_dir)
    print("GPU Enabled:", use_gpu)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load input CSV ----
    df = pd.read_csv(args.csv)
    print("Loaded rows:", len(df))

    df["shape"] = df["shape"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else s
    )
    df = df[df["shape"].apply(shape_ok)].copy()
    print("Rows after shape filter:", len(df))

    df["pathtodicom"] = df["directory"] + "/" + df["file_name"]
    df1 = df.reset_index(drop=True)

    # ---- EasyOCR (local models only) ----
    patch_torch_load()
    import easyocr
    reader = easyocr.Reader(
        args.langs,
        gpu=use_gpu,
        model_storage_directory=args.model_dir
    )

    # ---- process in chunks ----
    n = len(df1)
    n_chunks = math.ceil(n / args.chunk_size)
    print(f"Total rows: {n} | Chunks: {n_chunks}")

    for chunk_idx in range(n_chunks):
        start = chunk_idx * args.chunk_size
        stop = min(start + args.chunk_size, n)
        out_path = os.path.join(args.out_dir, f"{args.out_prefix}{chunk_idx+1}.csv")

        print(f"\n[Chunk {chunk_idx+1}/{n_chunks}] rows {start}-{stop-1}")
        chunk = df1.iloc[start:stop].copy()

        texts = []
        for i, row in chunk.iterrows():
            texts.append(ocr_from_dicom(row["pathtodicom"], reader))

        chunk["dicomtext"] = texts
        chunk.to_csv(out_path, index=False)
        print("Saved:", out_path)

        del chunk, texts
        gc.collect()

    print("DONE.")

if __name__ == "__main__":
    main()