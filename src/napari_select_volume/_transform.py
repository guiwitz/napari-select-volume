"""Crop and rotate a 3-D volume using SimpleITK (multi-threaded C++)."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ._shapes_utils import RectangleInfo


def numpy_to_sitk(volume: np.ndarray, scale: tuple[float, ...]) -> sitk.Image:
    """Convert a numpy volume (Z, Y, X) with napari-style *scale* to sitk.Image.

    ``scale`` is expected in array-axis order (z, y, x).
    """
    image = sitk.GetImageFromArray(volume)
    # SimpleITK uses (x, y, z) ordering
    image.SetSpacing((float(scale[2]), float(scale[1]), float(scale[0])))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection(np.eye(3).flatten().tolist())
    return image


def crop_and_rotate(
    sitk_image: sitk.Image,
    rect_info: RectangleInfo,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Crop the sub-volume defined by *rect_info* and rotate it axis-aligned.

    The rectangle drawn in the displayed 2-D plane is extruded along the
    perpendicular axis (full extent).  The output volume is rotated so that the
    rectangle edges are axis-aligned.

    Parameters
    ----------
    sitk_image : sitk.Image
        Input 3-D volume.
    rect_info : RectangleInfo
        Rectangle geometry as returned by ``get_rectangle_info``.

    Returns
    -------
    result : np.ndarray
        Cropped + rotated volume in (Z, Y, X) numpy order.
    out_scale : (sz, sy, sx)
        Spacing for the output volume in numpy axis order.
    """
    spacing = sitk_image.GetSpacing()  # (sx, sy, sz)
    size = sitk_image.GetSize()  # (nx, ny, nz)

    # Map displayed_axes (which are numpy axes 0=Z,1=Y,2=X) to SimpleITK axes
    # numpy axis 0 → sitk axis 2 (z), numpy axis 1 → sitk axis 1 (y),
    # numpy axis 2 → sitk axis 0 (x)
    _np_to_sitk = {0: 2, 1: 1, 2: 0}

    ax0_sitk = _np_to_sitk[rect_info.displayed_axes[0]]
    ax1_sitk = _np_to_sitk[rect_info.displayed_axes[1]]
    perp_sitk = _np_to_sitk[rect_info.perp_axis]

    # --- Rotation angle ---
    angle = -rect_info.angle  # radians, CCW in the displayed plane

    # --- Physical centre of the rectangle ---
    # rect_info.center_2d is in *numpy data coordinates* for the two displayed axes.
    # Convert to physical (SimpleITK) coordinates.
    center_phys = [0.0, 0.0, 0.0]  # (x, y, z)
    center_phys[ax0_sitk] = float(rect_info.center_2d[0]) * spacing[ax0_sitk]
    center_phys[ax1_sitk] = float(rect_info.center_2d[1]) * spacing[ax1_sitk]
    # Perpendicular axis: use the midpoint of the full extent
    center_phys[perp_sitk] = (size[perp_sitk] * spacing[perp_sitk]) / 2.0

    # --- Build the forward rotation (we will invert it for the resampler) ---
    # SimpleITK Euler3DTransform rotations: (rx, ry, rz) around X, Y, Z axes
    # of the *physical* coordinate system.
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_phys)

    # rotation args: (angle_x, angle_y, angle_z)
    # The resampler maps output→input via T.  The display coordinate system
    # (numpy: Z,Y,X) is left-handed w.r.t. the SimpleITK physical system
    # (x,y,z), so the rotation sense is reversed – we use +angle (not −angle).
    rot_args = [0.0, 0.0, 0.0]
    rot_args[perp_sitk] = angle
    transform.SetRotation(*rot_args)

    # --- Output reference image ---
    # Output spacing = same as input
    out_spacing = list(spacing)

    # Output size: width and height of the rectangle (in pixels) along the
    # two displayed axes; full extent along the perpendicular axis.
    out_size = [0, 0, 0]
    out_size[ax0_sitk] = int(np.ceil(rect_info.height))
    out_size[ax1_sitk] = int(np.ceil(rect_info.width))
    out_size[perp_sitk] = size[perp_sitk]

    # Output origin: the physical position that maps to the output voxel (0,0,0).
    # The resampler evaluates  input_phys = T(output_phys)  for every output
    # voxel.  We need T(out_origin) == corner0_phys, therefore:
    #   out_origin = T⁻¹(corner0_phys)
    corner0_phys = [0.0, 0.0, 0.0]
    corner0_phys[ax0_sitk] = float(rect_info.corner0_2d[0]) * spacing[ax0_sitk]
    corner0_phys[ax1_sitk] = float(rect_info.corner0_2d[1]) * spacing[ax1_sitk]
    corner0_phys[perp_sitk] = 0.0

    inv_transform = transform.GetInverse()
    out_origin = list(inv_transform.TransformPoint(tuple(corner0_phys)))

    # Direction cosines = identity (output is axis-aligned)
    out_direction = np.eye(3).flatten().tolist()

    # --- Resample ---
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(out_size)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputOrigin(out_origin)
    resampler.SetOutputDirection(out_direction)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk_image.GetPixelID())

    result_sitk = resampler.Execute(sitk_image)
    result = sitk.GetArrayFromImage(result_sitk)

    # Output scale in numpy axis order (z, y, x)
    out_scale = (
        out_spacing[2],  # z
        out_spacing[1],  # y
        out_spacing[0],  # x
    )

    return result, out_scale
