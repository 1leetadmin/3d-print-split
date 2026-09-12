"""Turn a binary voxel mask into a clean, watertight, correctly-oriented
surface mesh. Shared by the bulk color-split pipeline and the interactive
click-to-extract pipeline so both get the same correctness guarantees.
"""
from __future__ import annotations

from collections import Counter

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


def weld_local_topology_defects(
    mesh: trimesh.Trimesh,
    max_iterations: int = 5,
    max_bad_vertices: int = 4000,
    relative_defect_scale: float = 0.002,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Repair isolated non-manifold/boundary edges caused by tiny
    near-duplicate vertices (e.g. a spurious sub-millimeter sliver facet
    left by a slicer's per-triangle color-paint export, reusing one real
    edge but with a near-duplicate third corner), without touching the
    rest of the mesh's own geometry -- which can legitimately be just as
    densely tessellated nearby, so a global vertex weld at any tolerance
    that bridges these gaps would silently destroy real detail elsewhere.

    Only bad edges shorter than ``relative_defect_scale`` times the mesh's
    own bounding-box diagonal are treated as candidate duplicate-vertex
    artifacts, and only their endpoint vertices are ever considered for
    merging. This matters because a single defect can also orphan a
    normal-length edge alongside its tiny ones (e.g. a spurious sliver
    triangle re-using one real, ordinary-length mesh edge) -- lumping that
    into the same weld pass, or sizing the merge tolerance off the largest
    bad edge instead of the defect's own tiny ones, would incorrectly fuse
    distant, unrelated vertices together. A genuinely large hole is
    deliberately left alone here (its edges exceed the scale check) so the
    caller's own hole-filling/voxel fallback -- not a false weld that would
    silently distort the model -- handles it instead. Iterates because
    collapsing one small cluster can occasionally expose a new bad edge one
    hop over.

    Returns the repaired mesh (unchanged if there was nothing to do, or if
    the defect looks too widespread to be this kind of local issue -- the
    caller's own fallback then takes over) plus a boolean mask, aligned to
    the input mesh's original face order, marking which original faces
    survived (a defect's sliver triangles can become degenerate and get
    dropped once their vertices are merged).
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    kept_mask = np.ones(len(faces), dtype=bool)

    bbox_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    max_defect_edge_length = relative_defect_scale * bbox_diagonal

    for _ in range(max_iterations):
        working = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        edge_counts = Counter(tuple(e) for e in working.edges_sorted)
        bad_edges = [e for e, c in edge_counts.items() if c != 2]
        if not bad_edges:
            break

        tiny_bad_edges = [
            (a, b)
            for a, b in bad_edges
            if np.linalg.norm(vertices[a] - vertices[b]) <= max_defect_edge_length
        ]
        if not tiny_bad_edges:
            break  # remaining bad edges look like genuine gaps, not duplicates

        bad_vertex_ids = sorted({v for edge in tiny_bad_edges for v in edge})
        if len(bad_vertex_ids) > max_bad_vertices:
            break  # too widespread for a local fix; let the caller fall back

        tolerance = 2.0 * max(
            float(np.linalg.norm(vertices[a] - vertices[b])) for a, b in tiny_bad_edges
        )

        n = len(bad_vertex_ids)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(vertices[bad_vertex_ids[i]] - vertices[bad_vertex_ids[j]]) <= tolerance:
                    root_i, root_j = find(i), find(j)
                    if root_i != root_j:
                        parent[root_i] = root_j

        remap = np.arange(len(vertices))
        roots: dict[int, int] = {}
        for i, vertex_id in enumerate(bad_vertex_ids):
            root = find(i)
            remap[vertex_id] = roots.setdefault(root, vertex_id)

        faces = remap[faces]
        nondegenerate = (
            (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
        )
        if not nondegenerate.all():
            kept_indices = np.where(kept_mask)[0]
            kept_mask[kept_indices[~nondegenerate]] = False
        faces = faces[nondegenerate]

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), kept_mask


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
