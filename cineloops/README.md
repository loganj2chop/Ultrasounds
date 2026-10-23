# Kidney Ultrasound Cine Pipeline

Finds kidney ultrasound cine loops in a DICOM collection and converts each one to NIfTI, with a per-frame kidney segmentation.

```
tag-scraper CSV ─► 1. find cines ─► 2. OCR ─► 3. filter kidney ─► 4. cine → NIfTI
```

Each step is a Python script plus a `run_*.sh` runner. Edit the paths at the top of each runner, then run it. The Python scripts themselves need no edits.

| Step | Runner | Script | Input | Output |
|---|---|---|---|---|
| 1 | `run_find_cines.sh` | `find_cines.py` | Tag-scraper CSV | Cine-only CSV |
| 2 | `run_ocr.sh` | `ocr_dicom.py` | Cine-only CSV | OCR chunk CSVs (`<prefix>1.csv`, `<prefix>2.csv`, …) |
| 3 | `run_filter_kidney.sh` | `filter_kidney.py` | OCR chunk CSVs | Kidney-only CSV |
| 4 | `run_cine_to_nifti.sh` | `segment_cine_to_nifti.py` | Kidney-only CSV | Image + mask NIfTIs, manifest CSV |

---

## Requirements

Python 3.9+ with:

```
numpy pandas pydicom torch easyocr scikit-image SimpleITK segmentation-models-pytorch
```

- **GPU:** recommended for steps 2 and 4. Both fall back to CPU, but they are much slower.
- **Step 2 model files:** EasyOCR needs its model files in the `--model-dir` folder. On a machine without internet access, copy the `.EasyOCR` folder over first.
- **Step 4 model weights:** step 4 needs the trained kidney UNet weights (`model_pretrained.pth`).

---

## Step 1: Find cine loops

Keeps only the rows whose `shape` column describes a multi-frame (cine) image.

```bash
bash run_find_cines.sh
```

**Input:** the CSV from the DICOM tag scraper. It needs a `shape` column, plus `directory` and `file_name` for the later steps.

**Rule:**

| Shape | Treated as | Kept? |
|---|---|---|
| `(108, 899, 1442)` | frames × H × W (grayscale) | yes |
| `(300, 480, 480, 3)` | frames × H × W × channels | yes |
| `(800, 800, 3)` | single color image | no |
| `(800, 800)` | single grayscale image | no |
| `(1, …)` | one frame | no |

**Runner settings:**
- `CSV`, `OUT_CSV`: input and output paths.
- `MIN_FRAMES` (default 2): raise it to drop very short clips.

**Output:** the same columns as the input, cine rows only.

---

## Step 2: OCR the burned-in labels

Reads the first frame of each DICOM and runs EasyOCR on it to capture the sonographer's on-screen text (e.g. "RT KIDNEY SAG").

```bash
bash run_ocr.sh
```

**Input:** the cine CSV from step 1. DICOM paths are built as `directory + "/" + file_name`, so they must be local or mounted paths.

**Runner settings:**
- `CSV`, `OUT_DIR`: input CSV and output directory.
- `OUT_PREFIX`: output file prefix.
- `MODEL_DIR`: EasyOCR model folder.
- `CHUNK_SIZE` (default 1000): rows per output file.
- Add `--no-gpu` to the command for a CPU run.

**Output:** `<OUT_DIR>/<OUT_PREFIX>1.csv`, `<OUT_PREFIX>2.csv`, …, one per chunk. Each has the input columns plus `pathtodicom` and `dicomtext`. Writing in chunks means a crash only loses the current chunk.

**Notes:**
- **Unreadable files:** a file that can't be read or OCR'd gets empty `dicomtext` and the run continues.
- **4-channel images:** shapes with 4 channels (`…, 4`) are skipped here even though step 1 keeps them.

---

## Step 3: Filter to kidney images

Counts label keywords in the OCR text and keeps single-side kidney images.

```bash
bash run_filter_kidney.sh
```

**Input:** the OCR chunk CSVs from step 2. Pass them with a quoted glob, e.g. `"/path/choponlyocr*.csv"`, and they're combined automatically.

**Keyword patterns** (whole words, case-insensitive):

| Column | Matches |
|---|---|
| `kidney` | kidney, kid, rk, lk |
| `bladder` | bladder, bla |
| `right` | right, rk, rt |
| `left` | left, lt, lk |
| `sagittal` | sagittal, sag, sg |
| `transverse` | transverse, trns, trans, trv |
| `liver` | liver, lvr |

**Kept if:** `kidney == 1` and `liver == 0` and `bladder == 0` and exactly one of `left`/`right` is 1.

**Output:** the kidney-only rows, with the keyword-count columns added. `sagittal` and `transverse` are counted but not filtered on, so they're available for selecting views later.

**Notes:**
- **Double counting:** `rk` and `lk` count toward both `kidney` and the side. "RK SAG" gives `kidney = 1` and `right = 1`.
- **Kidney named twice:** text that names the kidney twice (e.g. "RT KIDNEY … RK") gives `kidney = 2` and is excluded.

---

## Step 4: Cine to NIfTI with kidney segmentation

Reads each cine loop, applies the 2D kidney UNet to every frame, and saves the result as a pair of NIfTIs.

```bash
bash run_cine_to_nifti.sh
```

**Input:** the kidney-only CSV from step 3. The DICOM path is read from the `pathtodicom` column, set with `--path-col`.

**Per frame:**
1. Convert to grayscale (YBR/RGB are converted first).
2. Resize to 256 × 256.
3. Crop (10% top/bottom, 5% left/right) and resize back to 256 × 256.
4. Run the UNet and threshold at 0.5.

This matches the preprocessing the model was trained with.

**Main arguments:**

| Argument | Default | Purpose |
|---|---|---|
| `--csv` | | Kidney-only CSV |
| `--output-dir` | | Where NIfTIs and the manifest go |
| `--model` | | UNet weights (`model_pretrained.pth`) |
| `--path-col` | `pathtodicom` | Column with DICOM paths |
| `--batch-size` | 32 | Frames per GPU batch |
| `--start`, `--end` | all rows | Row range, for splitting across jobs |
| `--overwrite` | off | Redo files that already exist |

**Output** (per DICOM, named `<study_id>_<dicom name>`, or `<row>_<dicom name>` if there's no `study_id` column):
- `<name>_image.nii.gz`: the preprocessed cine, 256 × 256 × frames, float32.
- `<name>_mask.nii.gz`: the binary kidney mask, same size, uint8.
- `manifest.csv`, or `manifest_<start>_<end>.csv` when run on a row range: status, original shape, frame count, frames with kidney, and frame time per file.

**Notes:**
- **Spacing:** pixel spacing is 1 × 1 (the image is resampled), and the z-spacing is the frame time in ms when the DICOM provides it.
- **Rerunning:** existing outputs are skipped unless `--overwrite` is set, so an interrupted run can just be started again.
- **Parallel jobs:** run several jobs on different `--start`/`--end` ranges.

---

## Full run

```bash
bash run_find_cines.sh       # tag CSV        -> cine CSV
bash run_ocr.sh              # cine CSV       -> OCR chunk CSVs
bash run_filter_kidney.sh    # OCR chunks     -> kidney CSV
bash run_cine_to_nifti.sh    # kidney CSV     -> NIfTIs + manifest
```

Before each step, make sure its runner's input path points at the previous step's output.