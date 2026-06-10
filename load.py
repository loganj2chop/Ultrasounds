#!/usr/bin/env python3
import os
import io
import gc
import time
import errno
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
from skimage.transform import resize
import segmentation_models_pytorch as smp
import pydicom
import gcsfs
from numpy.lib.format import open_memmap
import SimpleITK as sitk  # kept for reference

# ======================================================
# Defaults (override with CLI)
# ======================================================
DEFAULT_OUTPUT_DIR = "/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/"
DEFAULT_INPUT_CSV  = "/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/newfilteredforbi.csv"
DEFAULT_PARTS      = 5

# ======================================================
# Helpers
# ======================================================
def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

def _out(output_dir: str, *names):
    return os.path.join(output_dir, *names)

def resize_image(image, target_size):
    return resize(image, target_size, preserve_range=True, anti_aliasing=True)

def crop_image(image, top_percent, bottom_percent, left_percent, right_percent, target_size):
    """
    Crop fixed margins as a percent of height/width, then resize to target_size.
    This keeps the image 'whole' (just cropped), no masking.
    """
    height, width = image.shape[-2], image.shape[-1]
    top_crop = int(height * top_percent)
    bottom_crop = int(height * bottom_percent)
    left_crop = int(width * left_percent)
    right_crop = int(width * right_percent)
    cropped_image = image[..., top_crop:height-bottom_crop, left_crop:width-right_crop]
    resized_image = resize_image(cropped_image, target_size)
    return resized_image

def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert 3-channel arrays (RGB/YBR) to grayscale using BT.601 luma."""
    if arr.ndim == 3 and arr.shape[-1] == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(arr.dtype)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(arr.dtype)
    return arr

def chunk_bounds(n_items: int, n_chunks: int):
    """Yield (start, end) index pairs partitioning range(n_items) into n_chunks (end exclusive)."""
    if n_chunks <= 0:
        yield (0, n_items)
        return
    base = n_items // n_chunks
    rem = n_items % n_chunks
    start = 0
    for i in range(n_chunks):
        extra = 1 if i < rem else 0
        end = start + base + extra
        yield (start, end)
        start = end

# ======================================================
# Model
# ======================================================
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    )
    model.load_state_dict(
        torch.load(
            "/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/model_pretrained.pth",
            map_location="cpu",
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device

# ======================================================
# GCS + local DICOM reader with NFS retry handling
# ======================================================
_gcsfs = None

def _get_gcsfs():
    global _gcsfs
    if _gcsfs is None:
        _gcsfs = gcsfs.GCSFileSystem(token="google_default")
    return _gcsfs

def _gs_to_fs_key(gs_path: str) -> str:
    return gs_path[len("gs://"):]

def dcmread_local_with_retries(path: str, attempts: int = 5, base_sleep: float = 0.5):
    """Open a local/NFS DICOM with retries for 'stale file handle' errors."""
    last_err = None
    for i in range(attempts):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path, "rb", buffering=0) as f:
                return pydicom.dcmread(f, force=True)
        except OSError as e:
            last_err = e
            if getattr(e, "errno", None) in (errno.ESTALE, 116, errno.EIO):
                sleep = base_sleep * (2 ** i)
                print(f"[Retry {i+1}/{attempts}] Stale handle on {path}, retrying in {sleep:.2f}s")
                time.sleep(sleep)
                continue
            raise
    raise last_err

def dcmread_any(path: str):
    """Read a DICOM from local disk or gs:// path with NFS-safe retries."""
    if isinstance(path, str) and path.startswith("gs://"):
        fs = _get_gcsfs()
        with fs.open(_gs_to_fs_key(path), "rb") as f:
            return pydicom.dcmread(f, force=True)
    else:
        return dcmread_local_with_retries(path)

# ======================================================
# Processing: write each part incrementally via memmap
# ======================================================
def process_and_save_partition(
    df_part: pd.DataFrame,
    output_dir: str,
    prefix_base: str,
    part_idx: int,
    model,
    device,
    target_size=(256, 256),
):
    """
    Process df_part sequentially and write outputs directly to .npy files using open_memmap.
    Now saves:
      - cropped images: fold{fold}_{split}_image_{part}.npy
      - kidney masks:   fold{fold}_{split}_mask_{part}.npy
    """
    n = len(df_part)
    H, W = target_size

    # Prepare .npy files for incremental writing
    img_path = _out(output_dir, f"{prefix_base}_image_{part_idx}.npy")
    msk_path = _out(output_dir, f"{prefix_base}_mask_{part_idx}.npy")
    ids_path = _out(output_dir, f"{prefix_base}_ids_{part_idx}.csv")

    image_mmap = open_memmap(img_path, mode="w+", dtype="float32", shape=(n, H, W))
    mask_mmap  = open_memmap(msk_path, mode="w+", dtype="float32", shape=(n, H, W))

    ids_rows = []
    i = 0
    for _, row in df_part.iterrows():
        try:
            dcmfile_path = row["dcmfile"]
            ds = dcmread_any(dcmfile_path)

            pixel_array = ds.pixel_array
            pixel_array = to_grayscale(pixel_array)
            if pixel_array.ndim == 3:  # e.g., F x H x W
                pixel_array = pixel_array[0]

            # Resize and crop to focus on region of interest, but keep image whole
            processed_image = resize_image(pixel_array, target_size).astype(np.float32)
            cropped_resized_image = crop_image(
                processed_image,
                top_percent=0.1,
                bottom_percent=0.1,
                left_percent=0.05,
                right_percent=0.05,
                target_size=target_size,
            ).astype(np.float32)

            # Prepare tensor for segmentation model
            image_tensor = torch.tensor(
                cropped_resized_image, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(device)

            if image_tensor.max().item() > 1.0:
                image_tensor = image_tensor / 255.0

            with torch.no_grad():
                output = model(image_tensor)
                mask = torch.sigmoid(output).cpu().numpy().squeeze(0).squeeze(0).astype(np.float32)
                mask_cleaned = (mask > 0.5).astype(np.float32)

            # Save full cropped image + binary mask
            image_mmap[i, :, :] = cropped_resized_image
            mask_mmap[i, :, :]  = mask_cleaned

            ids_rows.append(
                [row["study_id"], row["dcmfile"], row["Bad"], row["new-old"], row["function"]]
            )
            i += 1

            # Free per-iteration
            del image_tensor, output, mask, mask_cleaned, cropped_resized_image, processed_image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if i % 250 == 0:
                print(f"  wrote {i}/{n} rows to {prefix_base}_image/mask_{part_idx}.npy")

        except Exception as e:
            print(f"Error processing file {row.get('dcmfile', '<missing dcmfile>')}: {e}")

    # Flush to disk
    del image_mmap, mask_mmap
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save CSV (small memory footprint)
    ids_df = pd.DataFrame(
        ids_rows, columns=["study_id", "dcmfile", "Bad", "new-old", "function"]
    )
    ids_df.to_csv(ids_path, index=False)
    del ids_df, ids_rows
    gc.collect()

# ======================================================
# Main: create folds but process only one fold selected
# ======================================================
def run_one_fold(df: pd.DataFrame, fold_index_1based: int, output_dir: str, parts: int):
    assert 1 <= fold_index_1based <= 5, "--fold must be in 1..5"
    output_dir = ensure_outdir(output_dir)

    model, device = load_model()
    gkf = GroupKFold(n_splits=5)

    # Build all fold indices so we can select one
    folds = list(gkf.split(df, groups=df["study_id"]))
    train_idx, test_idx = folds[fold_index_1based - 1]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)
    H, W = (256, 256)

    print(f"== Processing only fold {fold_index_1based}/5 ==")
    print(f"  Train size: {len(df_train)} rows")
    print(f"  Test  size: {len(df_test)} rows")
    print(f"  Output dir: {output_dir}")
    print(f"  Chunks per split: {parts}")

    # ---- TRAIN split in parts (direct-to-disk) ----
    base_prefix = f"fold{fold_index_1based}_train"
    n_train = len(df_train)
    for part_no, (s, e) in enumerate(chunk_bounds(n_train, parts), start=1):
        if s == e:
            # empty chunk (can happen if n < parts)
            img_path = _out(output_dir, f"{base_prefix}_image_{part_no}.npy")
            msk_path = _out(output_dir, f"{base_prefix}_mask_{part_no}.npy")
            ids_p    = _out(output_dir, f"{base_prefix}_ids_{part_no}.csv")
            open_memmap(img_path, mode="w+", dtype="float32", shape=(0, H, W))
            open_memmap(msk_path, mode="w+", dtype="float32", shape=(0, H, W))
            pd.DataFrame(columns=["study_id", "dcmfile", "Bad", "new-old", "function"]).to_csv(
                ids_p, index=False
            )
            continue

        print(f"[TRAIN] Fold {fold_index_1based} part {part_no}: rows {s}..{e-1} ({e-s} items)")
        process_and_save_partition(df_train.iloc[s:e], output_dir, base_prefix, part_no, model, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- TEST split in parts (direct-to-disk) ----
    base_prefix = f"fold{fold_index_1based}_test"
    n_test = len(df_test)
    for part_no, (s, e) in enumerate(chunk_bounds(n_test, parts), start=1):
        if s == e:
            img_path = _out(output_dir, f"{base_prefix}_image_{part_no}.npy")
            msk_path = _out(output_dir, f"{base_prefix}_mask_{part_no}.npy")
            ids_p    = _out(output_dir, f"{base_prefix}_ids_{part_no}.csv")
            open_memmap(img_path, mode="w+", dtype="float32", shape=(0, H, W))
            open_memmap(msk_path, mode="w+", dtype="float32", shape=(0, H, W))
            pd.DataFrame(columns=["study_id", "dcmfile", "Bad", "new-old", "function"]).to_csv(
                ids_p, index=False
            )
            continue

        print(f"[TEST ] Fold {fold_index_1based} part {part_no}: rows {s}..{e-1} ({e-s} items)")
        process_and_save_partition(df_test.iloc[s:e], output_dir, base_prefix, part_no, model, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"✅ Done: fold {fold_index_1based} saved to {output_dir}")

# ======================================================
# CLI
# ======================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Process one GroupKFold fold at a time, writing in parts per split."
    )
    ap.add_argument("--fold", type=int, required=True, help="Which fold to run (1..5).")
    ap.add_argument("--csv", type=str, default=DEFAULT_INPUT_CSV, help="Input CSV path.")
    ap.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory (absolute)."
    )
    ap.add_argument(
        "--parts", type=int, default=DEFAULT_PARTS, help="Number of chunks per split (default 5)."
    )
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_outdir(args.output_dir)

    df = pd.read_csv(args.csv, encoding="ISO-8859-1")
    df = df.drop_duplicates(subset=["dcmfile"])

    # Sanity checks
    required_cols = {"study_id", "dcmfile", "Bad", "new-old", "function"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    run_one_fold(df, args.fold, outdir, args.parts)

if __name__ == "__main__":
    main()
