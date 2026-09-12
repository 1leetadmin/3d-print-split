"""Main window: a left pane always shows the untouched original model, a
right pane is where you click a colored part to select it, cut it away from
the rest with an auto-sized peg/socket connector, and see extracted parts
laid out beside the shrinking body. Undo/redo each cut, export when done.
"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree
from PySide6.QtGui import QColor, QPixmap, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.core.loaders.detect import SUPPORTED_EXTENSIONS
from app.core.project import BodyState, ExtractedPart, Project
from app.gui.workers import ExtractWorker, ProjectLoadWorker

HIGHLIGHT_COLOR = (255, 0, 255)  # magenta, unmissable against any model color


def _to_pyvista(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).flatten()
    return pv.PolyData(vertices, cells)


def _color_icon(rgb: tuple[int, int, int]) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


def _layout_offsets(body: BodyState, parts: list[ExtractedPart]) -> list[np.ndarray]:
    """Translation offset per part, laying extracted parts out in a row to
    the right of the body's own bounding box so they're clearly separated
    and easy to inspect, instead of overlapping where they were cut from.
    """
    if len(body.vertices) == 0:
        body_extent_x = 0.0
        cursor_x = 0.0
    else:
        body_extent_x = float(body.vertices[:, 0].max() - body.vertices[:, 0].min())
        cursor_x = float(body.vertices[:, 0].max())
    margin = max(0.15 * body_extent_x, 5.0)
    cursor_x += margin

    offsets = []
    for part in parts:
        if len(part.vertices) == 0:
            offsets.append(np.zeros(3))
            continue
        part_min_x = float(part.vertices[:, 0].min())
        part_max_x = float(part.vertices[:, 0].max())
        dx = cursor_x - part_min_x
        offsets.append(np.array([dx, 0.0, 0.0]))
        cursor_x += (part_max_x - part_min_x) + margin
    return offsets


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STL Color Splitter")
        self.resize(1500, 850)

        self.current_path: str | None = None
        self.project: Project | None = None
        self.load_worker: ProjectLoadWorker | None = None
        self.extract_worker: ExtractWorker | None = None

        self.pending_selection: np.ndarray | None = None
        self._last_clicked_face: int | None = None
        self._centroid_tree: cKDTree | None = None
        self._centroid_tree_body: BodyState | None = None

        self._build_ui()

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        outer = QSplitter()
        self.setCentralWidget(outer)

        panel = QWidget()
        layout = QVBoxLayout(panel)

        open_btn = QPushButton("Open model…")
        open_btn.clicked.connect(self.on_open)
        layout.addWidget(open_btn)

        layout.addWidget(QLabel("Fallback repair/remesh resolution:"))
        self.voxel_spin = QSpinBox()
        self.voxel_spin.setRange(20, 500)
        self.voxel_spin.setValue(120)
        self.voxel_spin.setToolTip(
            "Only used if a cut can't be made by preserving the original surface\n"
            "(e.g. a badly broken source mesh, or an unusually irregular selection\n"
            "boundary) -- higher gives a finer fallback result but is slower."
        )
        layout.addWidget(self.voxel_spin)

        layout.addWidget(QLabel("Max colors detected:"))
        self.max_colors_spin = QSpinBox()
        self.max_colors_spin.setRange(1, 32)
        self.max_colors_spin.setValue(8)
        layout.addWidget(self.max_colors_spin)

        self.swap_rb_checkbox = QCheckBox("Swap R/B (legacy color-STL files)")
        layout.addWidget(self.swap_rb_checkbox)

        layout.addWidget(QLabel("Click tolerance (color-match sensitivity):"))
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 450)
        self.tolerance_spin.setValue(30)
        self.tolerance_spin.setToolTip(
            "How far a clicked patch is allowed to grow into neighboring colors.\n"
            "0 = exact color match only. 450 = grabs the whole connected surface."
        )
        self.tolerance_spin.valueChanged.connect(self.on_tolerance_changed)
        layout.addWidget(self.tolerance_spin)

        layout.addWidget(QLabel("Connector style:"))
        self.connector_style_combo = QComboBox()
        self.connector_style_combo.addItems(["Auto", "Round peg + hole", "Keyed (D-shaped)", "None"])
        layout.addWidget(self.connector_style_combo)

        layout.addWidget(QLabel("Connector size:"))
        self.connector_scale_spin = QDoubleSpinBox()
        self.connector_scale_spin.setRange(0.3, 3.0)
        self.connector_scale_spin.setSingleStep(0.1)
        self.connector_scale_spin.setValue(1.0)
        layout.addWidget(self.connector_scale_spin)

        layout.addWidget(QLabel("Extraction mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["Incremental (build up piece by piece)", "Independent (each from original)"]
        )
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        layout.addWidget(self.mode_combo)

        self.extract_btn = QPushButton("Extract selected part")
        self.extract_btn.setEnabled(False)
        self.extract_btn.clicked.connect(self.on_extract_clicked)
        layout.addWidget(self.extract_btn)

        undo_row = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.on_undo)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setEnabled(False)
        self.redo_btn.clicked.connect(self.on_redo)
        undo_row.addWidget(self.undo_btn)
        undo_row.addWidget(self.redo_btn)
        layout.addLayout(undo_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("Extracted parts:"))
        self.parts_list = QListWidget()
        layout.addWidget(self.parts_list)

        self.export_btn = QPushButton("Export all parts…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.on_export)
        layout.addWidget(self.export_btn)

        layout.addWidget(QLabel("Log:"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        panel.setMaximumWidth(360)
        outer.addWidget(panel)

        viewers = QSplitter()
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Original (untouched)"))
        self.left_plotter = QtInteractor(left_box)
        left_layout.addWidget(self.left_plotter)
        viewers.addWidget(left_box)

        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("Working copy -- click a part to select it"))
        self.right_plotter = QtInteractor(right_box)
        right_layout.addWidget(self.right_plotter)
        viewers.addWidget(right_box)

        viewers.setStretchFactor(0, 1)
        viewers.setStretchFactor(1, 1)
        outer.addWidget(viewers)
        outer.setStretchFactor(1, 1)

        self.right_plotter.enable_point_picking(
            callback=self.on_point_picked,
            picker="cell",
            left_clicking=True,
            show_message=False,
            show_point=True,
        )

        self.statusBar().showMessage("Open a .3mf, .obj, .ply, or color .stl file to begin")

    def log_message(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def _set_busy(self, busy: bool) -> None:
        self.centralWidget().setEnabled(not busy)

    # -- Loading -------------------------------------------------------------
    def on_open(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open colored model", "", f"Colored models ({exts});;All files (*)"
        )
        if not path:
            return
        self.current_path = path
        self.project = None
        self.pending_selection = None
        self._last_clicked_face = None
        self.parts_list.clear()
        self.extract_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.statusBar().showMessage(f"Loading {os.path.basename(path)} …")
        self.log_message(f"Loading {path}")
        self.progress_bar.setValue(0)

        independent = self.mode_combo.currentIndex() == 1
        self.load_worker = ProjectLoadWorker(
            path,
            voxels_along_longest=self.voxel_spin.value(),
            max_colors=self.max_colors_spin.value(),
            independent_mode=independent,
            swap_rb=self.swap_rb_checkbox.isChecked(),
        )
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.finished_ok.connect(self.on_loaded)
        self.load_worker.failed.connect(self.on_load_failed)
        self.load_worker.start()

    def on_load_progress(self, msg: str, frac: float) -> None:
        self.progress_bar.setValue(int(frac * 100))
        self.statusBar().showMessage(msg)

    def on_loaded(self, project: Project) -> None:
        self.project = project
        self.progress_bar.setValue(100)
        self.log_message("Model ready -- click a colored part to select it")
        self.statusBar().showMessage("Click a colored part of the model to select it")

        self.left_plotter.clear()
        body = project.master_body
        poly = _to_pyvista(body.vertices, body.faces)
        poly.cell_data["colors"] = body.face_colors
        self.left_plotter.add_mesh(poly, scalars="colors", rgb=True, show_scalar_bar=False)
        self.left_plotter.reset_camera()

        self.refresh_view()

    def on_load_failed(self, message: str) -> None:
        self.statusBar().showMessage("Load failed")
        self.log_message(f"ERROR loading file: {message}")
        QMessageBox.critical(self, "Could not load file", message)

    def on_mode_changed(self) -> None:
        if self.project is not None:
            self.project.independent_mode = self.mode_combo.currentIndex() == 1

    # -- Viewing / picking ----------------------------------------------------
    def refresh_view(self, keep_camera: bool = False) -> None:
        if self.project is None:
            return
        camera = self.right_plotter.camera_position if keep_camera else None
        self.right_plotter.clear()
        self.pending_selection = None
        self._last_clicked_face = None
        self.extract_btn.setEnabled(False)

        body = self.project.body
        poly = _to_pyvista(body.vertices, body.faces)
        poly.cell_data["colors"] = body.face_colors
        self.right_plotter.add_mesh(poly, scalars="colors", rgb=True, show_scalar_bar=False)

        for part, offset in zip(self.project.parts, _layout_offsets(body, self.project.parts)):
            part_poly = _to_pyvista(part.vertices + offset, part.faces)
            part_poly.cell_data["colors"] = part.face_colors
            self.right_plotter.add_mesh(part_poly, scalars="colors", rgb=True, show_scalar_bar=False)

        if camera is not None:
            self.right_plotter.camera_position = camera
        else:
            self.right_plotter.reset_camera()
        self._invalidate_centroid_cache()
        self.update_parts_list()

    def _invalidate_centroid_cache(self) -> None:
        self._centroid_tree = None
        self._centroid_tree_body = None

    def _centroid_tree_for_body(self, body: BodyState) -> cKDTree:
        if self._centroid_tree_body is not body:
            centroids = body.vertices[body.faces].mean(axis=1)
            self._centroid_tree = cKDTree(centroids)
            self._centroid_tree_body = body
        return self._centroid_tree

    def on_point_picked(self, point) -> None:
        if self.project is None:
            return
        body = self.project.body
        tree = self._centroid_tree_for_body(body)
        _dist, face_index = tree.query(np.asarray(point), k=1)
        face_index = int(face_index)
        if face_index >= len(body.faces):
            return  # a click landed on an already-extracted part, not the body
        self._last_clicked_face = face_index
        self._update_selection()

    def on_tolerance_changed(self) -> None:
        # Instant feedback: re-run the (cheap) flood-fill and re-highlight
        # without touching the actual cut, which only happens on Extract.
        if self._last_clicked_face is not None:
            self._update_selection()

    def _update_selection(self) -> None:
        if self.project is None or self._last_clicked_face is None:
            return
        body = self.project.body
        mask = self.project.select(self._last_clicked_face, tolerance=float(self.tolerance_spin.value()))
        self.pending_selection = mask
        self.log_message(f"Selected {int(mask.sum())} faces (tolerance {self.tolerance_spin.value()})")
        self.extract_btn.setEnabled(bool(mask.any()) and not bool(mask.all()))
        self._show_selection_highlight(body, mask)

    def _show_selection_highlight(self, body: BodyState, mask: np.ndarray) -> None:
        # Redraw the body dim, plus a bright highlight over the selected faces,
        # so the user can see exactly what "Extract" would cut out.
        camera = self.right_plotter.camera_position
        self.right_plotter.clear()
        poly = _to_pyvista(body.vertices, body.faces)
        poly.cell_data["colors"] = body.face_colors
        self.right_plotter.add_mesh(poly, scalars="colors", rgb=True, show_scalar_bar=False, opacity=0.35)
        if mask.any():
            highlight_faces = body.faces[mask]
            highlight_poly = _to_pyvista(body.vertices, highlight_faces)
            self.right_plotter.add_mesh(highlight_poly, color=HIGHLIGHT_COLOR, show_scalar_bar=False)
        for part, offset in zip(self.project.parts, _layout_offsets(body, self.project.parts)):
            part_poly = _to_pyvista(part.vertices + offset, part.faces)
            part_poly.cell_data["colors"] = part.face_colors
            self.right_plotter.add_mesh(part_poly, scalars="colors", rgb=True, show_scalar_bar=False)
        self.right_plotter.camera_position = camera
        self.right_plotter.render()

    # -- Extraction ------------------------------------------------------------
    def on_extract_clicked(self) -> None:
        if self.project is None or self.pending_selection is None:
            return
        style_map = {
            "Auto": "auto",
            "Round peg + hole": "round",
            "Keyed (D-shaped)": "keyed",
            "None": "none",
        }
        style = style_map[self.connector_style_combo.currentText()]
        name = f"part_{len(self.project.parts) + 1}"

        self._set_busy(True)
        self.statusBar().showMessage("Extracting…")
        self.extract_worker = ExtractWorker(
            self.project,
            self.pending_selection,
            name=name,
            connector_style=style,
            connector_scale=self.connector_scale_spin.value(),
        )
        self.extract_worker.progress.connect(self.on_load_progress)
        self.extract_worker.finished_ok.connect(self.on_extracted)
        self.extract_worker.failed.connect(self.on_extract_failed)
        self.extract_worker.start()

    def on_extracted(self, part: ExtractedPart) -> None:
        self._set_busy(False)
        connector_desc = f", {part.connector.style} connector" if part.connector else ", no connector"
        self.log_message(f"Extracted '{part.name}' ({len(part.faces)} faces{connector_desc})")
        self.statusBar().showMessage(f"Extracted '{part.name}'")
        self.export_btn.setEnabled(True)
        self.refresh_view()

    def on_extract_failed(self, message: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("Extraction failed")
        self.log_message(f"ERROR extracting: {message}")
        QMessageBox.critical(self, "Extraction failed", message)

    def update_parts_list(self) -> None:
        self.parts_list.clear()
        if self.project is None:
            return
        for part in self.project.parts:
            connector_desc = part.connector.style if part.connector else "no connector"
            item = QListWidgetItem(f"{part.name}  ({connector_desc})")
            item.setIcon(_color_icon(part.dominant_color))
            self.parts_list.addItem(item)
        self.undo_btn.setEnabled(self.project.can_undo)
        self.redo_btn.setEnabled(self.project.can_redo)
        self.export_btn.setEnabled(bool(self.project.parts))

    # -- Undo/redo -----------------------------------------------------------
    def on_undo(self) -> None:
        if self.project is None:
            return
        self.project.undo()
        self.log_message("Undo")
        self.refresh_view()

    def on_redo(self) -> None:
        if self.project is None:
            return
        self.project.redo()
        self.log_message("Redo")
        self.refresh_view()

    # -- Export -------------------------------------------------------------
    def on_export(self) -> None:
        if self.project is None or not self.project.parts:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return
        manifest_path = self.project.export_parts(out_dir)
        self.log_message(f"Exported {len(self.project.parts)} part(s) to {out_dir}")
        self.statusBar().showMessage(f"Exported to {out_dir}")
        QMessageBox.information(
            self,
            "Export complete",
            f"Wrote {len(self.project.parts)} part(s) plus the remaining body to:\n{out_dir}\n\n"
            f"See {os.path.basename(manifest_path)} for details.",
        )
