"""Turn a binary voxel mask into a clean, watertight, correctly-oriented
surface mesh. Shared by the bulk color-split pipeline and the interactive
click-to-extract pipeline so both get the same correctness guarantees.
"""
from __future__ import annotations

import numpy as np
import trimesh
from skimage import measure


def voxel_mask_to_watertight_mesh(
    mask: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct one watertight solid from a boolean voxel mask.

    ``mask`` is a dense boolean array in voxel-grid index space; ``transform``
    is the (4, 4) grid-index -> world-space affine used elsewhere in the
    pipeline. Returns (vertices, faces) in world space.

    A color region is frequently many disconnected islands, and marching
    cubes can leave a handful of non-manifold edges where two islands meet
    at a single ambiguous "pinch point" cell (a well-known topological
    ambiguity of the algorithm). Splitting into connected components and
    fixing normals/orientation per component -- each an unambiguous,
    individually watertight shell -- then rejoining resolves it: rejoined
    components get their own vertex copies at the old pinch point instead of
    one edge shared by both, which is what makes the combined result
    manifold again.
    """
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    verts_idx, faces, _normals, _values = measure.marching_cubes(
        padded.astype(np.float32), level=0.5
    )
    verts_idx -= 1.0  # undo the padding offset, back to caller's grid index space

    homogeneous = np.hstack([verts_idx, np.ones((len(verts_idx), 1))])
    world_verts = (transform @ homogeneous.T).T[:, :3]

    raw_mesh = trimesh.Trimesh(
        vertices=world_verts, faces=np.asarray(faces, dtype=np.int64), process=False
    )

    components = raw_mesh.split(only_watertight=False)
    if len(components) == 0:
        components = [raw_mesh]
    for component in components:
        trimesh.repair.fix_normals(component, multibody=False)
        if component.volume < 0:
            component.invert()
    merged = components[0] if len(components) == 1 else trimesh.util.concatenate(components)

    return np.asarray(merged.vertices), np.asarray(merged.faces, dtype=np.int64)


def transfer_colors_by_nearest_point(
    vertices: np.ndarray,
    faces: np.ndarray,
    sample_tree,
    sample_face_ids: np.ndarray,
    source_face_colors: np.ndarray,
) -> np.ndarray:
    """Per-face color for a reconstructed mesh, inherited from whichever
    point on a source mesh's surface sample is nearest each face's centroid.
    """
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    centroids = vertices[faces].mean(axis=1)
    _dist, nearest_idx = sample_tree.query(centroids, k=1)
    return source_face_colors[sample_face_ids[nearest_idx]]
