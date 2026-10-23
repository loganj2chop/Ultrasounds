#!/bin/bash
set -euo pipefail

# ============================
# RUN CONFIG — edit these
# ============================
OCR_CSVS="/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/choponlyocr*.csv"
OUT_CSV="/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/choponlyocr_kidney.csv"

python filter_kidney.py \
  --csv "$OCR_CSVS" \
  --out-csv "$OUT_CSV"