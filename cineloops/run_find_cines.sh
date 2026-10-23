#!/bin/bash
set -euo pipefail

# ============================
# RUN CONFIG — edit these
# ============================
CSV="/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/CHOPmag33d.csv"
OUT_CSV="/mnt/arcus/lab/project/New_ultrasounds/newmodelclevland/CHOPmag33d_cines.csv"
MIN_FRAMES=2

python find_cines.py \
  --csv "$CSV" \
  --out-csv "$OUT_CSV" \
  --min-frames "$MIN_FRAMES"