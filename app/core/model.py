"""Data model for a colored triangle mesh."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ColoredMesh:
    """A triangle mesh with one RGB color per face.

    Coordinates are in the same units/frame as the source file (typically mm).
    """

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int64, indices into vertices
    face_colors: np.ndarray  # (M, 3) uint8 RGB, one color per face
    source_path: str = ""
    # (M,) int, which original source object each face came from -- e.g. one
    # id per <object> in a multi-object 3MF/OBJ. None means the file is (or
    # was loaded as) a single continuous mesh with no natural object split.
    # When present, splitting keeps each object's own geometry independent
    # instead of merging touching objects into one mesh, which avoids the
    # non-manifold seams two objects placed face-to-face would otherwise
    # create.
    component_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        self.face_colors = np.asarray(self.face_colors, dtype=np.uint8).reshape(-1, 3)
        if self.face_colors.shape[0] != self.faces.shape[0]:
            raise ValueError(
                f"face_colors has {self.face_colors.shape[0]} rows but there are "
                f"{self.faces.shape[0]} faces"
            )
        if self.component_ids is not None:
            self.component_ids = np.asarray(self.component_ids, dtype=np.int64).reshape(-1)
            if self.component_ids.shape[0] != self.faces.shape[0]:
                raise ValueError(
                    f"component_ids has {self.component_ids.shape[0]} rows but there are "
                    f"{self.faces.shape[0]} faces"
                )

    @property
    def unique_colors(self) -> np.ndarray:
        """Distinct RGB colors actually present, sorted for stable ordering."""
        colors = np.unique(self.face_colors, axis=0)
        return colors[np.lexsort(colors.T[::-1])]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


@dataclass
class ColorPart:
    """One color's watertight solid, ready to be printed on its own plate."""

    color_rgb: tuple[int, int, int]
    vertices: np.ndarray  # (N, 3) float64, world-space, same frame as source mesh
    faces: np.ndarray  # (M, 3) int64
    voxel_count: int

    @property
    def hex_color(self) -> str:
        r, g, b = self.color_rgb
        return f"{r:02x}{g:02x}{b:02x}"
