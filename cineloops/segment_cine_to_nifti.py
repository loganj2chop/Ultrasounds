import numpy as np
import pandas as pd
import torch
import pydicom
import SimpleITK as sitk
import segmentation_models_pytorch as smp
from skimage.transform import resize

# ======================================================
# Defaults (override with CLI)
# ======================================================
DEFAULT_OUTPUT_DIR = "/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/cine_nifti/"
DEFAULT_INPUT_CSV  = "/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/cine_loops.csv"
DEFAULT_MODEL      = "/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/model_pretrained.pth"
DEFAULT_PATH_COL   = "pathtodicom"
TARGET_SIZE        = (256, 256)

# ======================================================
# Helpers (same as 2D script)
# ======================================================
def resize_image(image, target_size):
    return resize(image, target_size, preserve_range=True, anti_aliasing=True)

def crop_image(image, top_percent, bottom_percent, left_percent, right_percent, target_size):
    height, width = image.shape[-2], image.shape[-1]
    top_crop = int(height * top_percent)
    bottom_crop = int(height * bottom_percent)
    left_crop = int(width * left_percent)
    right_crop = int(width * right_percent)
    cropped_image = image[..., top_crop:height-bottom_crop, left_crop:width-right_crop]
    return resize_image(cropped_image, target_size)

def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """BT.601 luma for (..., 3) arrays; works for (H,W,3) and (F,H,W,3)."""
    if arr.ndim in (3, 4) and arr.shape[-1] == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    return arr.astype(np.float32)

def dcmread_local_with_retries(path: str, attempts: int = 5, base_sleep: float = 0.5):
    last_err = None
    for i in range(attempts):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path, "rb", buffering=0) as f:
                ds = pydicom.dcmread(f, force=True)
                _ = ds.pixel_array  # decode while file is open
                return ds
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
    if isinstance(path, str) and path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem(token="google_default")
        with fs.open(path[len("gs://"):], "rb") as f:
            ds = pydicom.dcmread(f, force=True)
            _ = ds.pixel_array
            return ds
    return dcmread_local_with_retries(path)

def convert_ybr_if_needed(ds, arr):
    """pydicom 2.x returns raw YBR for YBR_FULL/_422 color US; convert to RGB before luma."""
    pi = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if arr.shape[-1] == 3 and pi.startswith("YBR"):
        try:
            from pydicom.pixels import convert_color_space          # pydicom 3.x
        except ImportError:
            from pydicom.pixel_data_handlers.util import convert_color_space  # pydicom 2.x
        try:
            arr = convert_color_space(arr, pi, "RGB")
        except Exception:
            pass  # pydicom 3.x often already converted; leave as-is
    return arr

def get_frames(ds) -> np.ndarray:
    """Return float32 grayscale frames as (F, H, W) regardless of input layout."""
    arr = ds.pixel_array
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    arr = convert_ybr_if_needed(ds, arr)
    arr = to_grayscale(arr)
    if n_frames == 1 and arr.ndim == 2:   # single still image
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected pixel array shape after grayscale: {arr.shape}")
    return arr

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    img = resize_image(frame, TARGET_SIZE).astype(np.float32)
    img = crop_image(img, 0.1, 0.1, 0.05, 0.05, TARGET_SIZE).astype(np.float32)
    return img

# ======================================================
# Model
# ======================================================
def load_model(model_path):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), device

@torch.no_grad()
def segment_frames(images: np.ndarray, model, device, batch_size: int) -> np.ndarray:
    """images: (F, H, W) preprocessed. Returns uint8 masks (F, H, W)."""
    masks = np.zeros(images.shape, dtype=np.uint8)
    for s in range(0, len(images), batch_size):
        batch = images[s:s + batch_size].copy()
        # same per-image /255 rule as the 2D script
        maxes = batch.reshape(len(batch), -1).max(axis=1)
        batch[maxes > 1.0] /= 255.0
        t = torch.from_numpy(batch).unsqueeze(1).to(device)      # (B,1,H,W)
        prob = torch.sigmoid(model(t)).squeeze(1).cpu().numpy()  # (B,H,W)
        masks[s:s + batch_size] = (prob > 0.5).astype(np.uint8)
    return masks

# ======================================================
# NIfTI writing
# ======================================================
def write_nifti(arr: np.ndarray, path: str, frame_time_ms: float = None):
    """arr (F,H,W) -> NIfTI with size (W,H,F). Spacing 1x1 px; z spacing = frame time (ms) if known."""
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, float(frame_time_ms) if frame_time_ms else 1.0))
    sitk.WriteImage(img, path, useCompression=True)

def make_stem(row, path_col, idx):
    base = os.path.splitext(os.path.basename(str(row[path_col])))[0]
    if "study_id" in row.index and pd.notna(row["study_id"]):
        return f"{row['study_id']}_{base}"
    return f"{idx:06d}_{base}"

# ======================================================
# Main
# ======================================================
def main():
    ap = argparse.ArgumentParser(description="Per-frame kidney segmentation of US cine loops -> NIfTI.")
    ap.add_argument("--csv", default=DEFAULT_INPUT_CSV)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--path-col", default=DEFAULT_PATH_COL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--start", type=int, default=0, help="First row (for splitting across jobs).")
    ap.add_argument("--end", type=int, default=None, help="Last row, exclusive.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv, encoding="ISO-8859-1")
    df = df.drop_duplicates(subset=[args.path_col]).reset_index(drop=True)
    df = df.iloc[args.start:args.end]
    print(f"Rows to process: {len(df)}  (rows {args.start}..{args.end})")

    model, device = load_model(args.model)
    print(f"Device: {device}")

    manifest = []
    for idx, row in df.iterrows():
        dcm_path = row[args.path_col]
        stem = make_stem(row, args.path_col, idx)
        img_out = os.path.join(args.output_dir, f"{stem}_image.nii.gz")
        msk_out = os.path.join(args.output_dir, f"{stem}_mask.nii.gz")
        rec = {args.path_col: dcm_path, "image_nifti": img_out, "mask_nifti": msk_out}
        if "study_id" in row.index:
            rec["study_id"] = row["study_id"]

        if not args.overwrite and os.path.exists(img_out) and os.path.exists(msk_out):
            rec["status"] = "skipped_exists"
            manifest.append(rec)
            continue

        try:
            t0 = time.time()
            ds = dcmread_any(dcm_path)
            frames = get_frames(ds)                                   # (F,H,W)
            images = np.stack([preprocess_frame(f) for f in frames])  # (F,256,256)
            masks = segment_frames(images, model, device, args.batch_size)

            frame_time = getattr(ds, "FrameTime", None)
            write_nifti(images.astype(np.float32), img_out, frame_time)
            write_nifti(masks, msk_out, frame_time)

            rec.update(
                status="ok",
                orig_shape=str(ds.pixel_array.shape),
                n_frames=len(frames),
                frames_with_kidney=int((masks.reshape(len(masks), -1).sum(1) > 0).sum()),
                frame_time_ms=frame_time,
            )
            print(f"[{idx}] {stem}: {len(frames)} frames, "
                  f"{rec['frames_with_kidney']} with kidney, {time.time()-t0:.1f}s")
            del ds, frames, images, masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            rec.update(status=f"error: {e}")
            print(f"[{idx}] ERROR {dcm_path}: {e}")

        manifest.append(rec)

    suffix = f"_{args.start}_{args.end}" if (args.start or args.end is not None) else ""
    man_path = os.path.join(args.output_dir, f"manifest{suffix}.csv")
    pd.DataFrame(manifest).to_csv(man_path, index=False)
    print(f"✅ Done. Manifest: {man_path}")

if __name__ == "__main__":
    main()