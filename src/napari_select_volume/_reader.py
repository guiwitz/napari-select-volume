"""Fast DICOM volume reader using SimpleITK."""

from __future__ import annotations

import os
from pathlib import Path

import SimpleITK as sitk
import numpy as np


def napari_get_reader(path):
    """Return a reader function if *path* is a directory containing DICOM files."""
    if isinstance(path, list):
        path = path[0]

    path = str(path)

    if not os.path.isdir(path):
        return None

    # Quick check: at least one .dcm file (case-insensitive)
    has_dcm = any(
        f.lower().endswith(".dcm") for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )
    if not has_dcm:
        # SimpleITK can also detect DICOM files without .dcm extension
        # Try the GDCM series reader as a fallback
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
        if not series_ids:
            return None

    return _read_dicom_volume


def _read_dicom_volume(path):
    """Read a DICOM series from a directory and return as napari layer data."""
    if isinstance(path, list):
        path = path[0]
    path = str(path)

    reader = sitk.ImageSeriesReader()
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
    if not series_ids:
        raise ValueError(f"No DICOM series found in {path}")

    # Use the first series
    filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_ids[0])
    reader.SetFileNames(filenames)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image)  # (Z, Y, X) order

    # SimpleITK spacing is (x, y, z); napari scale needs (z, y, x) to match array axis order
    sx, sy, sz = image.GetSpacing()
    scale = (sz, sy, sx)

    # SimpleITK origin is (x, y, z); translate to (z, y, x)
    ox, oy, oz = image.GetOrigin()
    translate = (oz, oy, ox)

    meta = {
        "scale": scale,
        "translate": translate,
        "name": Path(path).name,
        "metadata": {
            "sitk_spacing": image.GetSpacing(),
            "sitk_origin": image.GetOrigin(),
            "sitk_direction": image.GetDirection(),
            "series_id": series_ids[0],
        },
    }

    return [(volume, meta, "image")]
