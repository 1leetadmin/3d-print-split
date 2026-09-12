"""Main window: open a colored model, preview it, split it into per-color
solids, and export each as its own printable STL.
"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
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

from app.core.export import export_parts
from app.core.loaders.detect import SUPPORTED_EXTENSIONS
from app.core.model import ColoredMesh, ColorPart
from app.gui.workers import LoadWorker, SplitWorker


def _to_pyvista(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).flatten()
    return pv.PolyData(vertices, cells)


def _color_icon(rgb: tuple[int, int, int]) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STL Color Splitter")
        self.resize(1200, 800)

        self.colored_mesh: ColoredMesh | None = None
        self.current_path: str | None = None
        self.parts: list[ColorPart] = []
        self.part_actors: list = []
        self.load_worker: LoadWorker | None = None
        self.split_worker: SplitWorker | None = None

        self._build_ui()

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter()
        self.setCentralWidget(splitter)

        # Left control panel
        panel = QWidget()
        layout = QVBoxLayout(panel)

        open_btn = QPushButton("Open model…")
        open_btn.clicked.connect(self.on_open)
        layout.addWidget(open_btn)

        self.swap_rb_checkbox = QCheckBox("Swap R/B (legacy color-STL files)")
        layout.addWidget(self.swap_rb_checkbox)

        layout.addWidget(QLabel("Voxel resolution (voxels along longest axis):"))
        self.voxel_spin = QSpinBox()
        self.voxel_spin.setRange(20, 500)
        self.voxel_spin.setValue(120)
        layout.addWidget(self.voxel_spin)

        layout.addWidget(QLabel("Max color parts:"))
        self.max_colors_spin = QSpinBox()
        self.max_colors_spin.setRange(1, 32)
        self.max_colors_spin.setValue(8)
        layout.addWidget(self.max_colors_spin)

        self.split_btn = QPushButton("Split into parts")
        self.split_btn.setEnabled(False)
        self.split_btn.clicked.connect(self.on_split)
        layout.addWidget(self.split_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("Parts:"))
        self.parts_list = QListWidget()
        self.parts_list.itemChanged.connect(self.on_part_visibility_changed)
        layout.addWidget(self.parts_list)

        export_row = QHBoxLayout()
        self.export_btn = QPushButton("Export all parts…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.on_export)
        export_row.addWidget(self.export_btn)
        layout.addLayout(export_row)

        layout.addWidget(QLabel("Log:"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        panel.setMaximumWidth(360)
        splitter.addWidget(panel)

        # 3D viewer
        self.plotter = QtInteractor(splitter)
        splitter.addWidget(self.plotter)
        splitter.setStretchFactor(1, 1)

        self.statusBar().showMessage("Open a .3mf, .obj, .ply, or color .stl file to begin")

    def log_message(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    # -- Loading -------------------------------------------------------------
    def on_open(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open colored model", "", f"Colored models ({exts});;All files (*)"
        )
        if not path:
            return
        self.current_path = path
        self.split_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.parts = []
        self.parts_list.clear()
        self.statusBar().showMessage(f"Loading {os.path.basename(path)} …")
        self.log_message(f"Loading {path}")

        self.load_worker = LoadWorker(path, swap_rb=self.swap_rb_checkbox.isChecked())
        self.load_worker.finished_ok.connect(self.on_loaded)
        self.load_worker.failed.connect(self.on_load_failed)
        self.load_worker.start()

    def on_loaded(self, colored_mesh: ColoredMesh) -> None:
        self.colored_mesh = colored_mesh
        n_colors = len(colored_mesh.unique_colors)
        self.log_message(
            f"Loaded {len(colored_mesh.faces)} faces, {n_colors} distinct source color(s)"
        )
        self.statusBar().showMessage(
            f"Loaded {os.path.basename(self.current_path or '')} "
            f"({len(colored_mesh.faces)} faces, {n_colors} colors)"
        )
        self.split_btn.setEnabled(True)

        self.plotter.clear()
        poly = _to_pyvista(colored_mesh.vertices, colored_mesh.faces)
        poly.cell_data["colors"] = colored_mesh.face_colors
        self.plotter.add_mesh(poly, scalars="colors", rgb=True, show_scalar_bar=False)
        self.plotter.reset_camera()

    def on_load_failed(self, message: str) -> None:
        self.statusBar().showMessage("Load failed")
        self.log_message(f"ERROR loading file: {message}")
        QMessageBox.critical(self, "Could not load file", message)

    # -- Splitting -------------------------------------------------------------
    def on_split(self) -> None:
        if self.colored_mesh is None:
            return
        self.split_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Splitting into color parts…")

        self.split_worker = SplitWorker(
            self.colored_mesh,
            voxels_along_longest=self.voxel_spin.value(),
            max_colors=self.max_colors_spin.value(),
        )
        self.split_worker.progress.connect(self.on_split_progress)
        self.split_worker.finished_ok.connect(self.on_split_done)
        self.split_worker.failed.connect(self.on_split_failed)
        self.split_worker.start()

    def on_split_progress(self, msg: str, frac: float) -> None:
        self.progress_bar.setValue(int(frac * 100))
        self.statusBar().showMessage(msg)

    def on_split_done(self, parts: list[ColorPart]) -> None:
        self.parts = parts
        self.split_btn.setEnabled(True)
        self.export_btn.setEnabled(bool(parts))
        self.progress_bar.setValue(100)
        self.log_message(f"Split into {len(parts)} part(s)")
        self.statusBar().showMessage(f"Split into {len(parts)} part(s)")

        self.plotter.clear()
        self.parts_list.clear()
        self.part_actors = []
        for part in parts:
            poly = _to_pyvista(part.vertices, part.faces)
            actor = self.plotter.add_mesh(
                poly, color=[c / 255 for c in part.color_rgb], show_scalar_bar=False
            )
            self.part_actors.append(actor)

            item = QListWidgetItem(f"#{part.hex_color}  ({part.voxel_count} voxels)")
            item.setIcon(_color_icon(part.color_rgb))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.parts_list.addItem(item)
        self.plotter.reset_camera()

    def on_split_failed(self, message: str) -> None:
        self.split_btn.setEnabled(True)
        self.statusBar().showMessage("Split failed")
        self.log_message(f"ERROR splitting: {message}")
        QMessageBox.critical(self, "Split failed", message)

    def on_part_visibility_changed(self, item: QListWidgetItem) -> None:
        row = self.parts_list.row(item)
        if 0 <= row < len(self.part_actors):
            self.part_actors[row].SetVisibility(item.checkState() == Qt.Checked)
            self.plotter.render()

    # -- Export -------------------------------------------------------------
    def on_export(self) -> None:
        if not self.parts:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return

        selected = [
            part
            for part, row in zip(self.parts, range(self.parts_list.count()))
            if self.parts_list.item(row).checkState() == Qt.Checked
        ]
        if not selected:
            QMessageBox.warning(self, "Nothing selected", "Check at least one part to export.")
            return

        manifest_path = export_parts(selected, out_dir)
        self.log_message(f"Exported {len(selected)} part(s) to {out_dir}")
        self.statusBar().showMessage(f"Exported {len(selected)} part(s) to {out_dir}")
        QMessageBox.information(
            self,
            "Export complete",
            f"Wrote {len(selected)} STL file(s) plus a manifest to:\n{out_dir}\n\n"
            f"See {os.path.basename(manifest_path)} for per-part details.",
        )
