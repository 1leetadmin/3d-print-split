"""Background threads so loading/voxelizing never freezes the UI."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from app.core.loaders.detect import load_colored_mesh
from app.core.model import ColoredMesh
from app.core.split import split_by_color


class LoadWorker(QThread):
    finished_ok = Signal(object)  # ColoredMesh
    failed = Signal(str)

    def __init__(self, path: str, swap_rb: bool = False):
        super().__init__()
        self.path = path
        self.swap_rb = swap_rb

    def run(self) -> None:
        try:
            mesh = load_colored_mesh(self.path, swap_rb=self.swap_rb)
            self.finished_ok.emit(mesh)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(str(exc))


class SplitWorker(QThread):
    progress = Signal(str, float)
    finished_ok = Signal(object)  # list[ColorPart]
    failed = Signal(str)

    def __init__(self, colored_mesh: ColoredMesh, voxels_along_longest: int, max_colors: int):
        super().__init__()
        self.colored_mesh = colored_mesh
        self.voxels_along_longest = voxels_along_longest
        self.max_colors = max_colors

    def run(self) -> None:
        try:
            parts = split_by_color(
                self.colored_mesh,
                voxels_along_longest=self.voxels_along_longest,
                max_colors=self.max_colors,
                progress_cb=lambda msg, frac: self.progress.emit(msg, frac),
            )
            self.finished_ok.emit(parts)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(str(exc))
