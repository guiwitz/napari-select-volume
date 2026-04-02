"""Geometry utilities for analysing rectangles drawn on napari Shapes layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RectangleInfo:
    """Parsed information about a rectangle drawn in one 2-D slice view."""

    angle: float  # rotation angle in radians (CCW from first displayed axis)
    center_2d: np.ndarray  # (2,) centre in the displayed plane (data coords)
    width: float  # side length along the first edge (pixels)
    height: float  # side length along the second edge (pixels)
    perp_axis: int  # axis index (0, 1 or 2) perpendicular to the view
    displayed_axes: tuple[int, int]  # the two axes visible in the slice view
    corner0_2d: np.ndarray  # (2,) first corner in the displayed plane


def get_rectangle_info(
    corners_nd: np.ndarray,
    displayed_axes: tuple[int, int],
) -> RectangleInfo:
    """Extract geometric properties from a napari shapes rectangle.

    Parameters
    ----------
    corners_nd : (4, ndim) array
        Corner coordinates as stored in ``shapes_layer.data[i]``.
        For a 3-D image viewed in 2-D, ndim == 3.
    displayed_axes : tuple of two ints
        The two axis indices currently displayed (e.g. ``(1, 2)`` for a Y-X
        view).  Obtained from ``viewer.dims.displayed``.

    Returns
    -------
    RectangleInfo
        Dataclass with angle, centre, width, height, perp_axis, etc.
    """
    ax0, ax1 = displayed_axes

    # Project onto the displayed 2-D plane
    c = corners_nd[:, [ax0, ax1]].astype(float)  # (4, 2)

    # napari rectangles have corners ordered: P0-P1-P2-P3 counter-clockwise
    edge0 = c[1] - c[0]  # first edge vector
    edge1 = c[3] - c[0]  # adjacent edge vector

    width = float(np.linalg.norm(edge0))
    height = float(np.linalg.norm(edge1))

    # Angle of edge1 measured CCW from the *first displayed axis* direction
    angle = float(np.arctan2(edge1[1], edge1[0]))

    center_2d = c.mean(axis=0)
    corner0_2d = c[0].copy()

    # Perpendicular axis = the one not displayed
    all_axes = {0, 1, 2}
    perp_axis = (all_axes - set(displayed_axes)).pop()

    return RectangleInfo(
        angle=angle,
        center_2d=center_2d,
        width=width,
        height=height,
        perp_axis=perp_axis,
        displayed_axes=displayed_axes,
        corner0_2d=corner0_2d,
    )


def rotate_rectangle(
    corners_nd: np.ndarray,
    displayed_axes: tuple[int, int],
    angle_deg: float,
) -> np.ndarray:
    """Return *corners_nd* rotated by *angle_deg* around the rectangle centre.

    Only the two displayed-axis coordinates are rotated; the perpendicular
    axis coordinate is left unchanged.

    Parameters
    ----------
    corners_nd : (4, ndim) array
        Original rectangle corners (data coordinates).
    displayed_axes : tuple of two ints
        The two currently displayed axes.
    angle_deg : float
        Rotation angle in degrees (CCW in the displayed plane).

    Returns
    -------
    new_corners : (4, ndim) array
        Rotated rectangle corners.
    """
    ax0, ax1 = displayed_axes
    pts = corners_nd.astype(float).copy()

    # 2-D coordinates in the display plane
    c2d = pts[:, [ax0, ax1]]  # (4, 2)
    center = c2d.mean(axis=0)

    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t],
                     [sin_t,  cos_t]])

    rotated = (rot @ (c2d - center).T).T + center
    pts[:, ax0] = rotated[:, 0]
    pts[:, ax1] = rotated[:, 1]
    return pts
