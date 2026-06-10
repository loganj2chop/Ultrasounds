#!/usr/bin/env python3
# GS-only DICOM tag extractor. "directory" = DICOM series (parent prefix).

import os
import pandas as pd
import pydicom
import gcsfs
from pathlib import PurePosixPath

# ---- SET YOUR GS PREFIX HERE ----


#### Will need to re-tool this for local files, but this is a start for GS-only. ----
TAGS_PATH = "gs://scit643-mag3-f50cec36-def/Cleveland/"


GET_SHAPE = True  # set False to avoid decoding pixels

# ---- parse bucket & prefix (no gs:// for gcsfs) ----
rest = TAGS_PATH[len("gs://"):]
bucket, _, key_prefix = rest.partition("/")
if key_prefix and not key_prefix.endswith("/"):
    key_prefix += "/"
prefix_for_fs = f"{bucket}/{key_prefix}"  # e.g. "bucket/folder1/folder2/"

# ---- connect & list recursively ----
fs = gcsfs.GCSFileSystem(token="google_default")
print("Exists:", fs.exists(prefix_for_fs))
print("Listing some entries under prefix…")
for p in fs.ls(prefix_for_fs)[:5]:
    print(" -", p)

dcm_objects = fs.glob(f"{prefix_for_fs}**/*.dcm")  # recursive *.dcm

# ---- iterate and extract tags (same logic as your script) ----
tags = []
for obj in dcm_objects:
    # obj is like "bucket/key/.../file.dcm"
    key = obj.split("/", 1)[1]                         # strip "bucket/"
    rel = key[len(key_prefix):].lstrip("/")            # path relative to base GS prefix
    rel_parts = [p for p in rel.split("/") if p]
    if not rel_parts:
        continue

    subject = rel_parts[0]
    file_name = PurePosixPath(key).name
    parent_prefix = str(PurePosixPath(key).parent)     # bucket-relative
    parent_directory = f"gs://{bucket}/{parent_prefix}"

    try:
        ds = pydicom.dcmread(
            fs.open(obj, "rb"),
            stop_before_pixels=not GET_SHAPE,
            force=True
        )

        dicom_tags = {}
        for e in ds.iterall():
            tag = e.tag
            try:
                description = pydicom.datadict.dictionary_description(tag)
                des_value = e.repval.replace('"', "")
            except KeyError:
                description = "Privatetag"
                des_value = ''
            if description != "Pixel Data":
                dicom_tags["subject"] = subject
                dicom_tags["file_name"] = file_name
                if GET_SHAPE:
                    try:
                        img = ds.pixel_array
                        dicom_tags["shape"] = getattr(img, "shape", None)
                    except Exception:
                        print(f"Pixel Data not found for file: gs://{obj}")
                dicom_tags["directory"] = parent_directory
                dicom_tags[description] = des_value

        tags.append(dicom_tags)

    except Exception as err:
        print(f"Error processing gs://{obj}: {err}")

# ---- to DataFrame & CSV ----
tags_df = pd.DataFrame(tags).drop_duplicates()
tags_df.to_csv('clevealltags.csv', index=False)
print("Wrote clevealltags.csv  with", len(tags_df), "rows")