"""Background threads so loading/voxelizing/extracting never freezes the UI."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.core.loaders.detect import load_colored_mesh
from app.core.project import ExtractedPart, Project


class ProjectLoadWorker(QThread):
    progress = Signal(str, float)
    finished_ok = Signal(object)  # Project
    failed = Signal(str)

    def __init__(
        self,
        path: str,
        voxels_along_longest: int,
        max_colors: int,
        independent_mode: bool,
        swap_rb: bool = False,
    ):
        super().__init__()
        self.path = path
        self.voxels_along_longest = voxels_along_longest
        self.max_colors = max_colors
        self.independent_mode = independent_mode
        self.swap_rb = swap_rb

    def run(self) -> None:
        try:
            mesh = load_colored_mesh(self.path, swap_rb=self.swap_rb)
            project = Project(
                mesh,
                voxels_along_longest=self.voxels_along_longest,
                max_colors=self.max_colors,
                independent_mode=self.independent_mode,
                progress_cb=lambda msg, frac: self.progress.emit(msg, frac),
            )
            self.finished_ok.emit(project)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(str(exc))


class ExtractWorker(QThread):
    progress = Signal(str, float)
    finished_ok = Signal(object)  # ExtractedPart
    failed = Signal(str)

    def __init__(
        self,
        project: Project,
        selected_face_mask: np.ndarray,
        name: str,
        connector_style: str,
        connector_scale: float,
    ):
        super().__init__()
        self.project = project
        self.selected_face_mask = selected_face_mask
        self.name = name
        self.connector_style = connector_style
        self.connector_scale = connector_scale

    def run(self) -> None:
        try:
            part: ExtractedPart = self.project.extract(
                self.selected_face_mask,
                name=self.name,
                connector_style=self.connector_style,
                connector_scale=self.connector_scale,
                progress_cb=lambda msg, frac: self.progress.emit(msg, frac),
            )
            self.finished_ok.emit(part)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(str(exc))
