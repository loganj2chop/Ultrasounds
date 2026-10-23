#!/bin/bash
set -euo pipefail

# ============================
# RUN CONFIG — edit these
# ============================
CSV="/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/CHOPmag33d.csv"
OUT_DIR="/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland"
OUT_PREFIX="choponlyocr"
MODEL_DIR="/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/.EasyOCR"
CHUNK_SIZE=1000

python 3dOCR.py \
  --csv "$CSV" \
  --out-dir "$OUT_DIR" \
  --out-prefix "$OUT_PREFIX" \
  --model-dir "$MODEL_DIR" \
  --chunk-size "$CHUNK_SIZE"