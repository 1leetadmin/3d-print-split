"""Turn a labeled voxel grid into one watertight, printable solid per color.

Objects that came from distinct places in the source file (e.g. one <object>
per color/filament in a multi-object 3MF) are voxelized completely
independently of each other, using a shared color palette and voxel pitch so
their outputs are consistent, but never merged into one mesh first. Two such
objects are commonly placed touching or coincident with each other (that's
normal for a model that's already split one part per color) -- merging them
would create a non-manifold seam that confuses solid voxelization and can
silently discard most of an object's volume. A file with no such natural
split (a single mesh with painted per-face/vertex colors) is voxelized as
one group, exactly as it was before object-awareness was added.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import trimesh
from skimage import measure

from app.core.colors import quantize_face_colors
from app.core.model import ColorPart, ColoredMesh
from app.core.voxelize import ProgressCB, VoxelLabeling, choose_pitch, voxelize_and_label


def _part_for_label(
    label: int, palette: np.ndarray, labeling: VoxelLabeling
) -> Optional[ColorPart]:
    mask = labeling.voxel_labels == label
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        return None

    dense_color = np.zeros(labeling.grid_shape, dtype=bool)
    idx = labeling.voxel_indices[mask]
    dense_color[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # Pad by one empty voxel on every side so marching cubes always closes
    # the surface at the volume boundary instead of leaving it open.
    padded = np.pad(dense_color, 1, mode="constant", constant_values=False)

    verts_idx, faces, _normals, _values = measure.marching_cubes(
        padded.astype(np.float32), level=0.5
    )
    verts_idx -= 1.0  # undo the padding offset, back to labeling.grid index space

    homogeneous = np.hstack([verts_idx, np.ones((len(verts_idx), 1))])
    world_verts = (labeling.transform @ homogeneous.T).T[:, :3]

    raw_mesh = trimesh.Trimesh(
        vertices=world_verts, faces=np.asarray(faces, dtype=np.int64), process=False
    )

    # A color region is frequently many disconnected islands (separate
    # blobs of the same color), and marching cubes can leave a handful of
    # non-manifold edges where two islands meet at a single ambiguous
    # "pinch point" cell. Splitting into connected components and fixing
    # normals/orientation per component -- each of which is an
    # unambiguous, individually watertight shell -- then rejoining them
    # resolves that: rejoined components get their own vertex copies at
    # the old pinch point instead of one edge shared by both, which is
    # exactly what makes the combined result manifold again.
    components = raw_mesh.split(only_watertight=False)
    if len(components) == 0:
        components = [raw_mesh]
    for component in components:
        trimesh.repair.fix_normals(component, multibody=False)
        if component.volume < 0:
            component.invert()
    part_mesh = components[0] if len(components) == 1 else trimesh.util.concatenate(components)

    color = tuple(int(c) for c in palette[label])
    return ColorPart(
        color_rgb=color,
        vertices=np.asarray(part_mesh.vertices),
        faces=np.asarray(part_mesh.faces, dtype=np.int64),
        voxel_count=voxel_count,
    )


def split_by_color(
    colored_mesh: ColoredMesh,
    voxels_along_longest: int = 120,
    max_colors: int = 8,
    progress_cb: ProgressCB = None,
) -> list[ColorPart]:
    """Full pipeline: quantize colors globally, then voxelize + label + solid-
    reconstruct each source object (or the whole mesh, if it has no natural
    object split) independently, merging same-colored results into one part.
    """
    def report(msg: str, frac: float) -> None:
        if progress_cb is not None:
            progress_cb(msg, frac)

    report("Quantizing colors", 0.0)
    palette, face_labels = quantize_face_colors(colored_mesh.face_colors, max_colors=max_colors)
    pitch = choose_pitch(colored_mesh.vertices, voxels_along_longest)

    if colored_mesh.component_ids is not None:
        unique_ids = np.unique(colored_mesh.component_ids)
        face_groups = [np.where(colored_mesh.component_ids == cid)[0] for cid in unique_ids]
    else:
        face_groups = [np.arange(len(colored_mesh.faces))]
    n_components = len(face_groups)

    meshes_by_color: dict[tuple, list[trimesh.Trimesh]] = defaultdict(list)
    voxels_by_color: dict[tuple, int] = defaultdict(int)

    for i, face_idx in enumerate(face_groups):
        base = 0.05 + 0.9 * (i / n_components)
        span = 0.9 / n_components

        def comp_progress(msg: str, frac: float, base=base, span=span, i=i) -> None:
            report(f"[object {i + 1}/{n_components}] {msg}", base + span * frac)

        comp_colored = ColoredMesh(
            vertices=colored_mesh.vertices,
            faces=colored_mesh.faces[face_idx],
            face_colors=palette[face_labels[face_idx]],
        )
        _mesh, labeling = voxelize_and_label(
            comp_colored,
            max_colors=max_colors,
            progress_cb=comp_progress,
            palette=palette,
            pitch=pitch,
        )
        for label in range(len(palette)):
            part = _part_for_label(label, palette, labeling)
            if part is not None:
                meshes_by_color[part.color_rgb].append(
                    trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
                )
                voxels_by_color[part.color_rgb] += part.voxel_count

    parts: list[ColorPart] = []
    for color, meshes in meshes_by_color.items():
        merged = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        parts.append(
            ColorPart(
                color_rgb=color,
                vertices=np.asarray(merged.vertices),
                faces=np.asarray(merged.faces, dtype=np.int64),
                voxel_count=voxels_by_color[color],
            )
        )

    report("Done", 1.0)
    return parts
