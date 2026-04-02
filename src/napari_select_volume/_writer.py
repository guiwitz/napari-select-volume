"""Writers for Zarr v3 (sharded/tiled) and BigTIFF output."""

from __future__ import annotations

import numpy as np


def save_zarr_v3(volume: np.ndarray, path: str, tile_n: int = 256) -> None:
    """Save *volume* as a Zarr v3 store with sharding.

    Each Z-plane is one shard.  Each shard is split into ``tile_n × tile_n``
    inner chunks (tiles).

    Parameters
    ----------
    volume : (Z, Y, X) ndarray
    path : str or Path
        Destination ``.zarr`` directory.
    tile_n : int
        Tile edge length in pixels.
    """
    import zarr
    from zarr.codecs import BloscCodec, BytesCodec, ShardingCodec

    nz, ny, nx = volume.shape

    # Each shard covers one full Z-plane → shard shape = (1, ny, nx)
    # Inner tile (chunk within shard) = (1, tile_n, tile_n)
    sharding_codec = ShardingCodec(
        chunk_shape=(1, tile_n, tile_n),
        codecs=[
            BytesCodec(),
            BloscCodec(cname="zstd", clevel=5),
        ],
    )

    z = zarr.open(
        path,
        mode="w",
        shape=(nz, ny, nx),
        chunks=(1, ny, nx),  # shard shape
        dtype=volume.dtype,
        zarr_format=3,
        codecs=[sharding_codec],
    )
    z[:] = volume


def save_zarr_ngio(volume: np.ndarray, path: str, tile_n: int = 256) -> None:
    """Save *volume* as an OME-Zarr using ngio with multi-scale pyramid.

    Parameters
    ----------
    volume : (Z, Y, X) ndarray
    path : str or Path
        Destination ``.zarr`` directory.
    tile_n : int
        Tile edge length in pixels.
    """
    import ngio

    nz, ny, nx = volume.shape

    ngio.create_ome_zarr_from_array(
        store=path,
        array=volume,
        axes_names=["z", "y", "x"],
        chunks=(1, tile_n, tile_n),
        shards=(1, ny, nx),
        overwrite=True,
    )


def save_tiff(volume: np.ndarray, path: str, tile_n: int = 256) -> None:
    """Save *volume* as a tiled BigTIFF with zstd compression.

    Parameters
    ----------
    volume : (Z, Y, X) ndarray
    path : str or Path
        Destination ``.tif`` / ``.tiff`` file.
    tile_n : int
        Tile edge length in pixels.
    """
    import tifffile

    tifffile.imwrite(
        path,
        volume,
        bigtiff=True,
        tile=(tile_n, tile_n),
        photometric="minisblack",
        compression="zstd",
    )
