"""Writers for Zarr v3 (sharded/tiled) and BigTIFF output."""

from __future__ import annotations
import numpy as np


def save_zarr_ngio(volume: np.ndarray, path: str, tile_size: int = 256) -> None:
    """Save *volume* as an OME-Zarr using ngio with multi-scale pyramid.

    Parameters
    ----------
    volume : (Z, Y, X) ndarray
    path : str or Path
        Destination ``.zarr`` directory.
    tile_size : int
        Tile edge length in pixels.
    """
    import ngio

    ngio.create_ome_zarr_from_array(
        store=path,
        array=volume,
        axes_names=["z", "y", "x"],
        levels=1,
        pixelsize=(1.0, 1.0),
        overwrite=True,
    )

def save_zarr_ngff(volume: np.ndarray, path: str, tile_size: int = 256) -> None:
    """Save *volume* as an OME-Zarr using ngff-zarr with multi-scale pyramid.

    Parameters
    ----------
    volume : (Z, Y, X) ndarray
    path : str or Path
        Destination ``.zarr`` directory.
    tile_size : int
        Tile edge length in pixels.
    """
    import ngff_zarr as nz

    chunks = tile_size  # tile_size or image size, whichever is smaller
    
    image = nz.to_ngff_image(volume,
                             dims=['z', 'y', 'x'],
                             scale={'z': 1.0, 'y': 1.0, 'x': 1.0},
                             translation={'z': 0.0, 'y': 0.0, 'x': 0.0})

    multiscales = nz.to_multiscales(image,
                                    scale_factors=[],
                                    chunks=(1, chunks, chunks),)
    
    nz.to_ngff_zarr(path, multiscales)


def save_tiff(volume: np.ndarray, path: str, tile_size: int = 256) -> None:
    """Save *volume* as a tiled BigTIFF with zstd compression.

    Parameters
    ----------
    volume : (Z, Y, X) ndarray
    path : str or Path
        Destination ``.tif`` / ``.tiff`` file.
    tile_size : int
        Tile edge length in pixels.
    """
    import tifffile

    tifffile.imwrite(
        path,
        volume,
        bigtiff=True,
        tile=(tile_size, tile_size),
        photometric="minisblack",
        compression="zstd",
    )
