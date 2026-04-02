"""Main Qt widget for the Select Volume plugin."""

from __future__ import annotations

import traceback

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from magicgui.widgets import create_widget

from ._shapes_utils import get_rectangle_info, rotate_rectangle
from ._transform import crop_and_rotate, numpy_to_sitk
from ._writer import save_tiff, save_zarr_v3, save_zarr_ngio

SHAPES_LAYER_NAME = "crop_region"


class SelectVolumeWidget(QWidget):
    """Napari widget for selecting, cropping, rotating and saving sub-volumes."""

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._cropped_volume: np.ndarray | None = None
        self._cropped_scale: tuple[float, float, float] | None = None
        # Base (unrotated) rectangle corners and the displayed axes when drawn
        self._base_corners: np.ndarray | None = None
        self._base_displayed_axes: tuple[int, int] | None = None
        self._shape_count: int = 0  # track number of shapes to detect new vs move

        self._build_ui()

        # Keep cell-layer combobox in sync with the viewer
        self.viewer.layers.events.inserted.connect(self._combo_image_layer.reset_choices)
        self.viewer.layers.events.removed.connect(self._combo_image_layer.reset_choices)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Image layer selector ---
        self._combo_image_layer = create_widget(annotation=napari.layers.Image, label='Image layer')
        self._combo_image_layer.reset_choices()
        layout.addWidget(self._combo_image_layer.native)

        # --- Shape layer button ---
        btn_add_shape = QPushButton("Add Shape Layer")
        btn_add_shape.setToolTip(
            "Add a Shapes layer and activate rectangle-drawing mode"
        )
        btn_add_shape.clicked.connect(self._add_shapes_layer)
        layout.addWidget(btn_add_shape)

        # --- Rotation slider (0–180°) ---
        row_rot = QHBoxLayout()
        row_rot.addWidget(QLabel("Rotation (°):"))
        self._slider_rot = QSlider(Qt.Orientation.Horizontal)
        self._slider_rot.setRange(0, 1800)  # 0.0–180.0 in tenths of degree
        self._slider_rot.setValue(0)
        self._slider_rot.setToolTip("Rotate the drawn rectangle (0–180°)")
        self._slider_rot.valueChanged.connect(self._on_rotation_changed)
        row_rot.addWidget(self._slider_rot)
        self._spin_rot = QDoubleSpinBox()
        self._spin_rot.setRange(0.0, 180.0)
        self._spin_rot.setDecimals(1)
        self._spin_rot.setSingleStep(0.5)
        self._spin_rot.setSuffix("°")
        self._spin_rot.setValue(0.0)
        self._spin_rot.valueChanged.connect(self._on_spin_rotation_changed)
        row_rot.addWidget(self._spin_rot)
        layout.addLayout(row_rot)

        # --- Crop & Rotate button ---
        btn_crop = QPushButton("Crop && Rotate")
        btn_crop.setToolTip(
            "Crop the volume to the drawn rectangle and rotate it axis-aligned"
        )
        btn_crop.clicked.connect(self._crop_and_rotate)
        layout.addWidget(btn_crop)

        # --- Separator ---
        layout.addSpacing(12)

        # --- Tile size ---
        row_tile = QHBoxLayout()
        row_tile.addWidget(QLabel("Tile size (N):"))
        self._spin_tile = QSpinBox()
        self._spin_tile.setRange(32, 4096)
        self._spin_tile.setSingleStep(32)
        self._spin_tile.setValue(256)
        row_tile.addWidget(self._spin_tile)
        layout.addLayout(row_tile)

        # --- Format selector ---
        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel("Format:"))
        self._combo_fmt = QComboBox()
        self._combo_fmt.addItems(["TIFF", "Zarr v3"])
        row_fmt.addWidget(self._combo_fmt)
        layout.addLayout(row_fmt)

        # --- Save button ---
        btn_save = QPushButton("Save Volume")
        btn_save.setToolTip("Save the cropped volume to disk")
        btn_save.clicked.connect(self._save_volume)
        layout.addWidget(btn_save)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _add_shapes_layer(self) -> None:
        """Add (or reuse) a Shapes layer with a default square in the view centre."""
        # Remove existing layer with the same name to avoid duplicates
        existing = [l for l in self.viewer.layers if l.name == SHAPES_LAYER_NAME]
        for l in existing:
            self.viewer.layers.remove(l)

        self._base_corners = None
        self._base_displayed_axes = None
        self._shape_count = 0
        self._slider_rot.setValue(0)

        # --- Build a default square centred in the current view ---
        displayed_axes = tuple(self.viewer.dims.displayed)  # e.g. (1, 2)
        ax0, ax1 = displayed_axes
        perp_axis = ({0, 1, 2} - set(displayed_axes)).pop()

        try:
            image_layer = self._get_active_image_layer()
            shape = np.array(image_layer.data.shape)  # (Z, Y, X)
        except RuntimeError:
            shape = np.array([100, 100, 100])

        dim0, dim1 = float(shape[ax0]), float(shape[ax1])
        half = min(dim0, dim1) / 2.0
        cx, cy = dim0 / 2.0, dim1 / 2.0
        perp_val = float(self.viewer.dims.current_step[perp_axis])

        # Corners: P0=top-left, P1=top-right, P2=bottom-right, P3=bottom-left
        # (top = min ax0, left = min ax1)
        corners_2d = np.array([
            [cx - half / 2, cy - half / 2],  # P0 top-left
            [cx - half / 2, cy + half / 2],  # P1 top-right
            [cx + half / 2, cy + half / 2],  # P2 bottom-right
            [cx + half / 2, cy - half / 2],  # P3 bottom-left
        ])

        default_rect = np.zeros((4, 3))
        default_rect[:, perp_axis] = perp_val
        default_rect[:, ax0] = corners_2d[:, 0]
        default_rect[:, ax1] = corners_2d[:, 1]

        shapes_layer = self.viewer.add_shapes(
            data=[default_rect],
            shape_type="rectangle",
            edge_color="yellow",
            face_color="transparent",
            edge_width=2,
            name=SHAPES_LAYER_NAME,
            ndim=3,
        )
        shapes_layer.mode = "direct"
        shapes_layer.events.data.connect(self._on_shape_data_changed)

        # Store the default square as the 0° base
        self._base_corners = default_rect.copy()
        self._base_displayed_axes = displayed_axes
        self._shape_count = 1

    def _on_shape_data_changed(self, _event=None) -> None:
        """Store the drawn rectangle as the 0° reference shape, or update the
        base when the user moves/resizes an existing rectangle.
        Only a single shape is allowed — newer shapes replace older ones."""
        try:
            shapes_layer = self._get_shapes_layer()
        except RuntimeError:
            return
        if len(shapes_layer.data) == 0:
            return

        current_count = len(shapes_layer.data)
        displayed_axes = tuple(self.viewer.dims.displayed)

        if current_count > 1:
            # Keep only the last drawn shape
            last = np.asarray(shapes_layer.data[-1]).copy()
            shapes_layer.events.data.disconnect(self._on_shape_data_changed)
            shapes_layer.data = [last]
            shapes_layer.events.data.connect(self._on_shape_data_changed)
            current_count = 1

        corners = np.asarray(shapes_layer.data[-1]).copy()

        if current_count > self._shape_count:
            # A new rectangle was drawn → this is the 0° base
            self._base_corners = corners
            self._base_displayed_axes = displayed_axes
            self._shape_count = current_count
            self._slider_rot.blockSignals(True)
            self._slider_rot.setValue(0)
            self._slider_rot.blockSignals(False)
            self._spin_rot.blockSignals(True)
            self._spin_rot.setValue(0.0)
            self._spin_rot.blockSignals(False)
        else:
            # Existing rectangle was moved/resized → reverse-rotate by current
            # slider angle to recover the new base, keeping the angle intact.
            current_angle = self._spin_rot.value()
            self._base_corners = rotate_rectangle(
                corners, displayed_axes, -current_angle
            )
            self._base_displayed_axes = displayed_axes

    def _on_rotation_changed(self, value: int) -> None:
        """Slider moved – update the spinbox and rotate the shape."""
        angle = value / 10.0
        self._spin_rot.blockSignals(True)
        self._spin_rot.setValue(angle)
        self._spin_rot.blockSignals(False)
        self._apply_rotation(angle)

    def _on_spin_rotation_changed(self, value: float) -> None:
        """Spinbox edited – update the slider and rotate the shape."""
        self._slider_rot.blockSignals(True)
        self._slider_rot.setValue(int(round(value * 10)))
        self._slider_rot.blockSignals(False)
        self._apply_rotation(value)

    def _apply_rotation(self, angle_deg: float) -> None:
        """Rotate the base rectangle by *angle_deg* and update the shapes layer."""
        if self._base_corners is None or self._base_displayed_axes is None:
            return
        try:
            shapes_layer = self._get_shapes_layer()
        except RuntimeError:
            return

        new_corners = rotate_rectangle(
            self._base_corners,
            self._base_displayed_axes,
            angle_deg,
        )

        # Temporarily disconnect to avoid re-triggering _on_shape_data_changed
        shapes_layer.events.data.disconnect(self._on_shape_data_changed)
        if len(shapes_layer.data) > 0:
            shapes_layer.data = [new_corners]
        shapes_layer.events.data.connect(self._on_shape_data_changed)

    def _crop_and_rotate(self) -> None:
        """Read the last drawn rectangle, crop and rotate the active image."""
        try:
            image_layer = self._get_active_image_layer()
            shapes_layer = self._get_shapes_layer()

            if len(shapes_layer.data) == 0:
                self._warn("No rectangle drawn yet. Draw one first.")
                return

            corners = np.asarray(shapes_layer.data[-1])  # (4, ndim)
            displayed_axes = tuple(self.viewer.dims.displayed)

            rect_info = get_rectangle_info(corners, displayed_axes)

            # Use the widget's rotation angle instead of the computed one
            rect_info.angle = np.deg2rad(self._spin_rot.value())

            # Convert to SimpleITK image
            scale = image_layer.scale  # (z, y, x) numpy order
            sitk_image = numpy_to_sitk(
                np.asarray(image_layer.data), tuple(scale)
            )

            result, out_scale = crop_and_rotate(sitk_image, rect_info)

            self._cropped_volume = result
            self._cropped_scale = out_scale

            self.viewer.add_image(
                result,
                scale=out_scale,
                name=f"{image_layer.name}_cropped",
            )
        except Exception as exc:
            self._warn(f"Crop & Rotate failed:\n{exc}\n\n{traceback.format_exc()}")

    def _save_volume(self) -> None:
        """Save the most recent cropped volume to disk."""
        if self._cropped_volume is None:
            self._warn("No cropped volume to save. Run Crop & Rotate first.")
            return

        fmt = self._combo_fmt.currentText()
        tile_n = self._spin_tile.value()

        if fmt == "Zarr v3":
            filt = "Zarr store (*.zarr)"
        else:
            filt = "TIFF files (*.tif *.tiff)"

        path, _ = QFileDialog.getSaveFileName(self, "Save Volume", "", filt)
        if not path:
            return

        try:
            if fmt == "Zarr v3":
                if not path.endswith(".zarr"):
                    path += ".zarr"
                save_zarr_ngio(self._cropped_volume, path, tile_n)
            else:
                if not path.endswith((".tif", ".tiff")):
                    path += ".tif"
                save_tiff(self._cropped_volume, path, tile_n)

            QMessageBox.information(self, "Saved", f"Volume saved to:\n{path}")
        except Exception as exc:
            self._warn(f"Save failed:\n{exc}\n\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_active_image_layer(self) -> napari.layers.Image:
        """Return the currently selected Image layer"""
        
        if self._combo_image_layer.value is not None:
            return self._combo_image_layer.value
        else:
            raise RuntimeError("No Image layer selected. Please select one from the dropdown.")

    def _get_shapes_layer(self):
        """Return the crop-region Shapes layer."""
        for layer in self.viewer.layers:
            if layer.name == SHAPES_LAYER_NAME:
                return layer
        raise RuntimeError(
            f"No Shapes layer named '{SHAPES_LAYER_NAME}' found. "
            "Click 'Add Shape Layer' first."
        )

    @staticmethod
    def _warn(msg: str) -> None:
        QMessageBox.warning(None, "Select Volume", msg)
