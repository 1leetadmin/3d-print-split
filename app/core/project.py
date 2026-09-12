"""In-memory session state for the interactive click-to-extract workflow:
one clean watertight "body" to click on, a list of finalized extracted
parts, and an undo/redo history of extraction operations.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import trimesh

from app.core.colors import quantize_face_colors
from app.core.extraction import ConnectorInfo, ExtractionResult, extract_region
from app.core.meshing import transfer_colors_by_nearest_point, voxel_mask_to_watertight_mesh
from app.core.model import ColoredMesh
from app.core.selection import build_face_adjacency_list, flood_fill_by_color
from app.core.voxelize import build_processed_trimesh, choose_pitch

ProgressCB = Optional[Callable[[str, float], None]]


@dataclass
class BodyState:
    vertices: np.ndarray
    faces: np.ndarray
    face_colors: np.ndarray


@dataclass
class ExtractedPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    face_colors: np.ndarray
    connector: Optional[ConnectorInfo]

    @property
    def dominant_color(self) -> tuple[int, int, int]:
        colors, counts = np.unique(self.face_colors, axis=0, return_counts=True)
        return tuple(int(c) for c in colors[np.argmax(counts)])


@dataclass
class _HistoryEntry:
    body_before: BodyState
    parts_before: list[ExtractedPart]
    body_after: BodyState
    parts_after: list[ExtractedPart]


def build_master_body(
    colored_mesh: ColoredMesh,
    voxels_along_longest: int = 120,
    max_colors: int = 8,
    progress_cb: ProgressCB = None,
) -> BodyState:
    """One-time voxelize + reconstruct of the whole model into a single
    clean watertight solid with the original colors (quantized down to at
    most ``max_colors``) carried over. Every later extraction cuts from
    this (or a piece of it), so it only needs to happen once per loaded
    file, however un-watertight the source mesh is. Quantizing colors here
    keeps the click flood-fill predictable against a bounded palette
    instead of a noisy near-continuous one.
    """
    def report(msg: str, frac: float) -> None:
        if progress_cb is not None:
            progress_cb(msg, frac)

    report("Cleaning mesh", 0.0)
    mesh = build_processed_trimesh(colored_mesh)
    pitch = choose_pitch(mesh.vertices, voxels_along_longest)

    report("Voxelizing", 0.2)
    voxel = mesh.voxelized(pitch=pitch).fill()
    dense = np.asarray(voxel.matrix, dtype=bool)
    transform = np.asarray(voxel.transform, dtype=np.float64)

    report("Reconstructing solid", 0.5)
    vertices, faces = voxel_mask_to_watertight_mesh(dense, transform)

    report("Coloring solid", 0.85)
    from scipy.spatial import cKDTree

    raw_face_colors = np.asarray(mesh.visual.face_colors)[:, :3].astype(np.uint8)
    palette, face_labels = quantize_face_colors(raw_face_colors, max_colors=max_colors)
    face_colors = palette[face_labels]

    n_samples = int(np.clip(len(mesh.faces) * 4, 2_000, 400_000))
    sample_points, sample_face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    tree = cKDTree(sample_points)
    colors = transfer_colors_by_nearest_point(vertices, faces, tree, sample_face_ids, face_colors)

    report("Done", 1.0)
    return BodyState(vertices=vertices, faces=faces, face_colors=colors)


class Project:
    """One loaded model's interactive extraction session.

    ``independent_mode``: when False (default), each extraction shrinks the
    body you click on next (build up an assembly piece by piece). When True,
    every extraction is cut from the pristine starting body regardless of
    what's already been extracted -- simpler, but two extracted parts won't
    know about each other's connectors.
    """

    def __init__(
        self,
        colored_mesh: ColoredMesh,
        voxels_along_longest: int = 120,
        max_colors: int = 8,
        independent_mode: bool = False,
        progress_cb: ProgressCB = None,
    ):
        self.original_mesh = colored_mesh
        self.voxels_along_longest = voxels_along_longest
        self.independent_mode = independent_mode

        self._master_body = build_master_body(
            colored_mesh, voxels_along_longest, max_colors, progress_cb
        )
        self.body: BodyState = self._master_body
        self.parts: list[ExtractedPart] = []

        self._undo_stack: list[_HistoryEntry] = []
        self._redo_stack: list[_HistoryEntry] = []

        self._adjacency_cache: Optional[list[list[int]]] = None
        self._adjacency_cache_body: Optional[BodyState] = None

    @property
    def master_body(self) -> BodyState:
        """The pristine, unmodified starting solid -- what "Show original" displays."""
        return self._master_body

    # -- selection -----------------------------------------------------------
    def adjacency_for_current_body(self) -> list[list[int]]:
        if self._adjacency_cache_body is not self.body:
            mesh = trimesh.Trimesh(vertices=self.body.vertices, faces=self.body.faces, process=False)
            self._adjacency_cache = build_face_adjacency_list(mesh)
            self._adjacency_cache_body = self.body
        return self._adjacency_cache

    def select(self, face_index: int, tolerance: float) -> np.ndarray:
        """Flood-fill selection on the body you'd currently click on."""
        adjacency = self.adjacency_for_current_body()
        return flood_fill_by_color(self.body.face_colors, adjacency, face_index, tolerance)

    # -- extraction ------------------------------------------------------------
    def extract(
        self,
        selected_face_mask: np.ndarray,
        name: str,
        connector_style: str = "auto",
        connector_scale: float = 1.0,
        progress_cb: ProgressCB = None,
    ) -> ExtractedPart:
        result: ExtractionResult = extract_region(
            self.body.vertices,
            self.body.faces,
            self.body.face_colors,
            selected_face_mask,
            voxels_along_longest=self.voxels_along_longest,
            connector_style=connector_style,
            connector_scale=connector_scale,
            progress_cb=progress_cb,
        )

        part = ExtractedPart(
            name=name,
            vertices=result.extracted_vertices,
            faces=result.extracted_faces,
            face_colors=result.extracted_face_colors,
            connector=result.connector,
        )
        new_body = BodyState(
            vertices=result.remainder_vertices,
            faces=result.remainder_faces,
            face_colors=result.remainder_face_colors,
        )

        body_before = self.body
        parts_before = list(self.parts)

        self.parts.append(part)
        if not self.independent_mode:
            self.body = new_body
        # In independent mode, self.body deliberately doesn't change: every
        # extraction is cut from the same starting point.

        self._undo_stack.append(
            _HistoryEntry(
                body_before=body_before,
                parts_before=parts_before,
                body_after=self.body,
                parts_after=list(self.parts),
            )
        )
        self._redo_stack.clear()
        return part

    # -- undo/redo -----------------------------------------------------------
    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> None:
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        self._redo_stack.append(entry)
        self.body = entry.body_before
        self.parts = list(entry.parts_before)

    def redo(self) -> None:
        if not self._redo_stack:
            return
        entry = self._redo_stack.pop()
        self._undo_stack.append(entry)
        self.body = entry.body_after
        self.parts = list(entry.parts_after)

    # -- export ---------------------------------------------------------------
    def export_parts(self, out_dir: str, include_remainder: bool = True) -> str:
        """Write each extracted part, and the current remaining body, as its
        own STL, plus a manifest.json. Returns the manifest path.
        """
        os.makedirs(out_dir, exist_ok=True)
        manifest: dict = {"parts": []}

        def write_piece(index: int, name: str, vertices, faces, connector: Optional[ConnectorInfo]):
            filename = f"part_{index:02d}_{name}.stl"
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.export(os.path.join(out_dir, filename), file_type="stl")
            entry = {
                "filename": filename,
                "name": name,
                "is_watertight": bool(mesh.is_watertight),
                "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
            }
            if connector is not None:
                entry["connector"] = {
                    "style": connector.style,
                    "radius_mm": connector.radius_mm,
                    "length_mm": connector.length_mm,
                }
            manifest["parts"].append(entry)

        for i, part in enumerate(self.parts, start=1):
            write_piece(i, part.name, part.vertices, part.faces, part.connector)

        if include_remainder and len(self.body.faces) > 0:
            write_piece(len(self.parts) + 1, "remaining_body", self.body.vertices, self.body.faces, None)

        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path
