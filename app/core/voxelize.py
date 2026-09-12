"""Voxelize a colored mesh and label each solid voxel with the color of the
nearest point on the original surface.

Painted color only lives on the mesh surface; propagating it inward by
nearest-surface-point is the same approach slicers use to decide which
extruder owns interior infill under a painted region.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from app.core.colors import quantize_face_colors
from app.core.model import ColoredMesh

ProgressCB = Optional[Callable[[str, float], None]]


@dataclass
class VoxelLabeling:
    palette: np.ndarray  # (K, 3) uint8
    voxel_indices: np.ndarray  # (V, 3) int, indices into the dense grid
    voxel_labels: np.ndarray  # (V,) int, index into palette
    transform: np.ndarray  # (4, 4) float, grid-index -> world coordinates
    grid_shape: tuple[int, int, int]
    pitch: float


def choose_pitch(vertices: np.ndarray, voxels_along_longest: int) -> float:
    mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    longest = float((maxs - mins).max())
    if longest <= 0:
        raise ValueError("Degenerate mesh: zero-size bounding box")
    return longest / max(int(voxels_along_longest), 8)


def build_processed_trimesh(colored_mesh: ColoredMesh) -> trimesh.Trimesh:
    """Merge/clean the mesh with trimesh while carrying face colors through."""
    mesh = trimesh.Trimesh(
        vertices=colored_mesh.vertices,
        faces=colored_mesh.faces,
        face_colors=colored_mesh.face_colors,
        process=True,
        validate=False,
    )
    return mesh


def _assign_to_palette(face_colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map each face color to its (expected exact) index in an existing
    palette, falling back to nearest-color for any that don't match exactly.
    """
    lut = {tuple(c): i for i, c in enumerate(palette)}
    labels = np.empty(len(face_colors), dtype=np.int64)
    unmatched = []
    for i, c in enumerate(face_colors):
        key = tuple(int(v) for v in c)
        idx = lut.get(key)
        if idx is None:
            unmatched.append(i)
        else:
            labels[i] = idx
    if unmatched:
        diffs = face_colors[unmatched].astype(np.int32)[:, None, :] - palette[None, :, :].astype(
            np.int32
        )
        dist = np.sum(diffs**2, axis=2)
        labels[unmatched] = np.argmin(dist, axis=1)
    return labels


def voxelize_and_label(
    colored_mesh: ColoredMesh,
    voxels_along_longest: int = 120,
    max_colors: int = 8,
    progress_cb: ProgressCB = None,
    palette: Optional[np.ndarray] = None,
    pitch: Optional[float] = None,
) -> tuple[trimesh.Trimesh, VoxelLabeling]:
    """Voxelize one mesh and label every solid voxel by nearest-surface color.

    If ``palette`` is given, faces are assigned to that fixed palette instead
    of being re-clustered -- used when processing one connected component of
    a larger model so every component shares the same color set. If ``pitch``
    is given, it overrides ``voxels_along_longest`` -- used for the same
    reason, so every component gets the same absolute voxel size.
    """
    def report(msg: str, frac: float) -> None:
        if progress_cb is not None:
            progress_cb(msg, frac)

    report("Cleaning mesh", 0.0)
    mesh = build_processed_trimesh(colored_mesh)
    if not mesh.is_watertight:
        # A non-watertight input can still be voxelized/filled reasonably,
        # but the caller should know results may have gaps.
        report(
            "Warning: input mesh is not watertight; voxel fill may be imperfect",
            0.05,
        )

    face_colors = np.asarray(mesh.visual.face_colors)[:, :3].astype(np.uint8)
    if palette is None:
        palette, face_labels = quantize_face_colors(face_colors, max_colors=max_colors)
    else:
        face_labels = _assign_to_palette(face_colors, palette)

    if pitch is None:
        pitch = choose_pitch(mesh.vertices, voxels_along_longest)

    report("Voxelizing", 0.15)
    voxel = mesh.voxelized(pitch=pitch)
    voxel = voxel.fill()
    dense = np.asarray(voxel.matrix, dtype=bool)
    transform = np.asarray(voxel.transform, dtype=np.float64)

    voxel_indices = np.argwhere(dense)
    if len(voxel_indices) == 0:
        raise ValueError("Voxelization produced no solid voxels; try a finer resolution")

    homogeneous = np.hstack([voxel_indices.astype(np.float64), np.ones((len(voxel_indices), 1))])
    world_points = (transform @ homogeneous.T).T[:, :3]

    # Exact point-to-triangle projection (trimesh.proximity.closest_point) is
    # too slow/memory-hungry once a mesh has hundreds of thousands of faces
    # and there are hundreds of thousands of voxels to label -- a real-world
    # painted model easily reaches both. A dense area-weighted surface sample
    # plus a KDTree nearest-neighbor lookup is dramatically cheaper and, since
    # voxelization itself already discretizes at a coarser scale, loses
    # negligible accuracy in practice.
    # Sample density should track surface/color complexity (face count), not
    # voxel count -- a simple low-poly mesh voxelized at high resolution
    # doesn't need hundreds of thousands of samples just because it produced
    # hundreds of thousands of voxels.
    n_samples = int(np.clip(len(mesh.faces) * 4, 2_000, 400_000))
    report(f"Sampling {n_samples} surface points for color lookup", 0.3)
    sample_points, sample_face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    tree = cKDTree(sample_points)

    report(f"Labeling {len(voxel_indices)} voxels by nearest surface color", 0.4)
    _dist, nearest_sample_idx = tree.query(world_points, k=1)
    triangle_ids = sample_face_ids[nearest_sample_idx]
    voxel_labels = face_labels[triangle_ids]

    report("Voxelization complete", 0.6)

    labeling = VoxelLabeling(
        palette=palette,
        voxel_indices=voxel_indices,
        voxel_labels=voxel_labels,
        transform=transform,
        grid_shape=dense.shape,
        pitch=pitch,
    )
    return mesh, labeling
