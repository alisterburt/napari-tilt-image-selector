from os import PathLike
from pathlib import Path
from typing import List, Tuple

import dask
import dask.array as da
import mdocfile
import mrcfile
import numpy as np
import pandas as pd
from thefuzz import process


def match_tilt_image_filenames(
    tilt_image_files: List[PathLike], mdoc_df: pd.DataFrame
) -> pd.DataFrame:
    """"""
    tilt_image_file_basenames = [Path(f).stem for f in tilt_image_files]
    mdoc_df["mdoc_tilt_image_basename"] = mdoc_df["sub_frame_path"].apply(
        lambda x: Path(str(x).split("\\")[-1]).stem
    )
    tilt_image_basename_to_full = {
        tilt_image_file_basenames[i]: tilt_image_files[i]
        for i in range(len(tilt_image_files))
    }
    matched_filenames = []
    for mdoc_tilt_image_filename in mdoc_df["mdoc_tilt_image_basename"]:
        match, _ = process.extractOne(
            query=mdoc_tilt_image_filename,
            choices=tilt_image_file_basenames,
        )
        matched_file = tilt_image_basename_to_full[match]
        matched_filenames.append(matched_file)
    mdoc_df["matched_filename"] = matched_filenames
    return mdoc_df


def get_ordered_tilt_images(mdoc_file, tilt_image_files):
    df = mdocfile.read(mdoc_file)
    df = match_tilt_image_filenames(tilt_image_files, mdoc_df=df)
    df = df.sort_values(by="tilt_angle", ascending=True)
    ordered_tilt_image_files = df["matched_filename"]
    return ordered_tilt_image_files


def get_image_shape(filename: PathLike) -> Tuple[int, int, int]:
    with mrcfile.open(filename, header_only=True) as mrc:
        nz, ny, nx = mrc.header.nz, mrc.header.ny, mrc.header.nx
    return int(nz), int(ny), int(nx)


def read_mrc(filename: PathLike):
    with mrcfile.open(filename) as mrc:
        data = mrc.data
    return data


def normalise_image(image):
    return (image - np.mean(image)) / np.std(image)


def lazy_tilt_series_from_tilt_images(tilt_image_files):
    sample = read_mrc(tilt_image_files[0])
    delayed_read_mrc = dask.delayed(read_mrc)
    delayed_arrays = [delayed_read_mrc(file) for file in tilt_image_files]
    dask_arrays = [
        da.from_delayed(delayed_array, shape=sample.shape, dtype=sample.dtype)
        for delayed_array in delayed_arrays
    ]
    lazy_tilt_series = da.stack(dask_arrays, axis=0)
    lazy_tilt_series = lazy_tilt_series.map_blocks(normalise_image)
    return lazy_tilt_series
