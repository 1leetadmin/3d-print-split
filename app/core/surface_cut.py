"""Surface-preserving cut: separate a selected patch of the ORIGINAL mesh
from the rest by capping the cut boundary with new triangles, instead of
re-voxelizing the whole model. Keeps essentially all of each part's surface
as the original geometry (full resolution) -- new triangles only appear
right at the seam and the connector.

This can fail on an irregular/non-manifold selection boundary (raises
``SurfaceCutFailed``); the caller is expected to fall back to the
voxel-based cut in that case, which is slower and blockier but always
succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mapbox_earcut as earcut
import numpy as np
import trimesh

N_SEGMENTS = 24


class SurfaceCutFailed(Exception):
    """The boundary topology is too irregular for a clean surface-preserving
    cut; caller should fall back to voxel-based cutting."""


@dataclass
class SurfaceCutConnector:
    style: str
    radius_mm: float
    length_mm: float
    center_world: np.ndarray
    axis_world: np.ndarray


@dataclass
class SurfaceCutResult:
    extracted_vertices: np.ndarray
    extracted_faces: np.ndarray
    remainder_vertices: np.ndarray
    remainder_faces: np.ndarray
    connector: Optional[SurfaceCutConnector]


def find_boundary_loops(mesh: trimesh.Trimesh, selected_mask: np.ndarray) -> list[np.ndarray]:
    """Ordered vertex-index loops along the boundary between selected and
    unselected faces. Raises SurfaceCutFailed if the boundary isn't a clean
    set of simple closed loops (e.g. a pixelated/speckled selection).
    """
    face_adjacency = mesh.face_adjacency
    face_adjacency_edges = mesh.face_adjacency_edges
    if len(face_adjacency) == 0:
        raise SurfaceCutFailed("mesh has no face adjacency data")

    a, b = face_adjacency[:, 0], face_adjacency[:, 1]
    boundary_mask = selected_mask[a] != selected_mask[b]
    edges = face_adjacency_edges[boundary_mask]
    if len(edges) == 0:
        raise SurfaceCutFailed("selection has no boundary")

    degree: dict[int, int] = {}
    adjacency: dict[int, list[int]] = {}
    for v1, v2 in edges:
        v1, v2 = int(v1), int(v2)
        degree[v1] = degree.get(v1, 0) + 1
        degree[v2] = degree.get(v2, 0) + 1
        adjacency.setdefault(v1, []).append(v2)
        adjacency.setdefault(v2, []).append(v1)

    if any(d != 2 for d in degree.values()):
        raise SurfaceCutFailed("boundary is not a set of simple loops")

    visited_edges: set[tuple[int, int]] = set()
    loops: list[np.ndarray] = []
    for v1, v2 in edges:
        v1, v2 = int(v1), int(v2)
        key = (min(v1, v2), max(v1, v2))
        if key in visited_edges:
            continue
        loop = [v1, v2]
        visited_edges.add(key)
        current = v2
        closed = False
        for _ in range(len(edges) + 1):
            next_vertex = None
            for cand in adjacency[current]:
                edge_key = (min(current, cand), max(current, cand))
                if edge_key not in visited_edges:
                    next_vertex = cand
                    break
            if next_vertex is None:
                raise SurfaceCutFailed("boundary loop did not close cleanly")
            edge_key = (min(current, next_vertex), max(current, next_vertex))
            visited_edges.add(edge_key)
            if next_vertex == loop[0]:
                closed = True
                break
            loop.append(next_vertex)
            current = next_vertex
        if not closed:
            raise SurfaceCutFailed("boundary loop did not close cleanly")
        loops.append(np.array(loop, dtype=np.int64))

    if len(visited_edges) != len(edges):
        raise SurfaceCutFailed("not all boundary edges were consumed by loop tracing")

    return loops


def best_fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    u_axis, v_axis, normal = vt[0], vt[1], vt[2]
    return centroid, normal, u_axis, v_axis


def _project(points: np.ndarray, centroid: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
    rel = points - centroid
    return np.stack([rel @ u_axis, rel @ v_axis], axis=1)


def _signed_area(points_2d: np.ndarray) -> float:
    x, y = points_2d[:, 0], points_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _unproject(points_2d: np.ndarray, centroid: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
    return centroid + points_2d[:, 0:1] * u_axis + points_2d[:, 1:2] * v_axis


def _earcut_single_ring(points_2d: np.ndarray) -> np.ndarray:
    """Triangulate one simple polygon (no holes). A deterministic,
    geometry-scaled jitter is applied for the topology decision only (the
    caller always maps results back to the real, unjittered coordinates via
    the returned index topology) -- exactly-collinear runs of points (common
    on a subdivided/faceted mesh boundary) and, when this is called on a
    hole-bridged polygon, the bridge's coincident double-back edge can
    otherwise confuse ear-clipping into skipping some boundary edges.
    """
    rng = np.random.default_rng(0)
    scale = float(np.ptp(points_2d, axis=0).max()) or 1.0
    jittered = points_2d + rng.uniform(-1, 1, size=points_2d.shape) * scale * 1e-7

    ring_ends = np.array([len(points_2d)], dtype=np.uint32)
    flat = earcut.triangulate_float64(jittered.astype(np.float64), ring_ends)
    triangles = np.asarray(flat, dtype=np.int64).reshape(-1, 3)

    n = len(points_2d)
    expected_edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    tri_edges = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            tri_edges.add((min(u, v), max(u, v)))
    if not expected_edges.issubset(tri_edges):
        raise SurfaceCutFailed("triangulation did not use every boundary edge")

    return triangles


def _triangulate_2d(outer_2d: np.ndarray, holes_2d: list[np.ndarray]) -> np.ndarray:
    """Triangulate ``outer_2d`` (a simple polygon, CCW) with zero or one hole
    (CW). Reduces a polygon-with-hole to a single simple ring by bridging the
    hole to its nearest outer vertex with a zero-width slit (a standard,
    well-tested technique), rather than relying on mapbox_earcut's own
    multi-ring hole-linking -- that has been observed to occasionally use
    fewer than every outer edge on a polygon with a hole and many
    exactly-collinear outer points (common on a subdivided/faceted mesh).
    """
    if not holes_2d:
        return _earcut_single_ring(outer_2d)
    if len(holes_2d) > 1:
        raise SurfaceCutFailed("more than one hole is not supported")

    hole_2d = holes_2d[0]
    n_o, n_h = len(outer_2d), len(hole_2d)
    diffs = outer_2d[:, None, :] - hole_2d[None, :, :]
    i, j = np.unravel_index(np.argmin(np.sum(diffs**2, axis=2)), (n_o, n_h))

    hole_seq = list(range(j, n_h)) + list(range(0, j + 1))
    combined_2d = np.vstack([outer_2d[: i + 1], hole_2d[hole_seq], outer_2d[i:]])
    # local index -> which original ring/position, for the caller to remap
    combined_positions = (
        [("outer", k) for k in range(i + 1)]
        + [("hole", k) for k in hole_seq]
        + [("outer", k) for k in range(i, n_o)]
    )

    tris_local = _earcut_single_ring(combined_2d)

    def remap(local_idx: int) -> tuple[str, int]:
        return combined_positions[local_idx]

    # Encode as a flat array of (is_hole, index) pairs the caller decodes;
    # simplest is to just return triangles already split by ring, so build
    # two integer codes here: outer local index unchanged, hole local index
    # offset by n_o (matching the convention callers already expect).
    out_tris = np.empty_like(tris_local)
    for a in range(tris_local.shape[0]):
        for b in range(3):
            kind, k = remap(int(tris_local[a, b]))
            out_tris[a, b] = k if kind == "outer" else n_o + k
    return out_tris


def _circle_points(
    centroid: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    radius: float,
    n_segments: int,
    flat_offset: Optional[float],
    clockwise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # earcut requires a hole ring's winding to be opposite the outer ring's;
    # the outer ring is always normalized to counter-clockwise below, so
    # holes default to clockwise.
    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    if clockwise:
        angles = -angles
    pts_2d = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    if flat_offset is not None:
        pts_2d[:, 0] = np.minimum(pts_2d[:, 0], flat_offset)
    pts_3d = _unproject(pts_2d, centroid, u_axis, v_axis)
    return pts_3d, pts_2d


def _build_tube_and_cap(
    rim_indices: list[int],
    tip_points_3d: np.ndarray,
    tip_center: np.ndarray,
    add_vertex,
) -> list[list[int]]:
    n = len(rim_indices)
    tip_indices = [add_vertex(p) for p in tip_points_3d]
    tip_center_idx = add_vertex(tip_center)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        r0, r1 = rim_indices[i], rim_indices[j]
        t0, t1 = tip_indices[i], tip_indices[j]
        faces.append([r0, r1, t1])
        faces.append([r0, t1, t0])
    for i in range(n):
        j = (i + 1) % n
        faces.append([tip_indices[i], tip_indices[j], tip_center_idx])
    return faces


def cut_mesh_preserving_surface(
    mesh: trimesh.Trimesh,
    selected_mask: np.ndarray,
    connector_style: str,
    connector_scale: float,
) -> SurfaceCutResult:
    loops = find_boundary_loops(mesh, selected_mask)

    def loop_perimeter(loop: np.ndarray) -> float:
        pts = mesh.vertices[loop]
        return float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))

    perimeters = [loop_perimeter(loop) for loop in loops]
    main_idx = int(np.argmax(perimeters))

    base_len = len(mesh.vertices)
    extracted_extra: list[np.ndarray] = []
    remainder_extra: list[np.ndarray] = []
    extracted_faces_new: list[list[int]] = []
    remainder_faces_new: list[list[int]] = []

    def add_extracted(p: np.ndarray) -> int:
        extracted_extra.append(p)
        return base_len + len(extracted_extra) - 1

    def add_remainder(p: np.ndarray) -> int:
        remainder_extra.append(p)
        return base_len + len(remainder_extra) - 1

    connector_info: Optional[SurfaceCutConnector] = None
    selected_centroid = mesh.vertices[mesh.faces[selected_mask]].reshape(-1, 3).mean(axis=0)
    remainder_centroid = mesh.vertices[mesh.faces[~selected_mask]].reshape(-1, 3).mean(axis=0)

    for i, loop in enumerate(loops):
        loop_points = mesh.vertices[loop]
        centroid, normal, u_axis, v_axis = best_fit_plane(loop_points)
        outer_2d = _project(loop_points, centroid, u_axis, v_axis)
        if _signed_area(outer_2d) < 0:
            loop = loop[::-1]
            outer_2d = outer_2d[::-1]

        make_connector = (i == main_idx) and connector_style != "none"

        if not make_connector:
            tris_local = _triangulate_2d(outer_2d, [])
            faces_global = loop[tris_local].tolist()
            extracted_faces_new.extend(faces_global)
            remainder_faces_new.extend(faces_global)
            continue

        radial = outer_2d - outer_2d.mean(axis=0)
        dist = np.linalg.norm(radial, axis=1)
        loop_radius_estimate = float(np.percentile(dist, 90))
        peg_radius = float(np.clip(0.35 * loop_radius_estimate * connector_scale, 1.0, 40.0))
        peg_length = float(np.clip(2.0 * peg_radius, 1.5, 30.0))
        clearance = max(0.15, 0.15 * peg_radius)
        socket_radius = peg_radius + clearance
        socket_depth = peg_length + 0.5

        resolved_style = connector_style
        if connector_style == "auto":
            cov = np.cov(radial.T)
            eigvals = np.linalg.eigvalsh(cov)
            aspect = float(np.sqrt(max(eigvals) / max(min(eigvals), 1e-9)))
            resolved_style = "keyed" if (aspect > 1.6 and peg_radius >= 2.0) else "round"

        flat_offset = peg_radius * 0.6 if resolved_style == "keyed" else None
        socket_flat_offset = socket_radius * 0.6 if resolved_style == "keyed" else None

        axis = remainder_centroid - selected_centroid
        if np.dot(axis, normal) < 0:
            normal = -normal

        hole_pts_3d, hole_pts_2d = _circle_points(centroid, u_axis, v_axis, peg_radius, N_SEGMENTS, flat_offset)
        socket_pts_3d, socket_pts_2d = _circle_points(
            centroid, u_axis, v_axis, socket_radius, N_SEGMENTS, socket_flat_offset
        )

        # extracted side: annulus with a peg-shaped hole, plus the protruding peg
        annulus_tris = _triangulate_2d(outer_2d, [hole_pts_2d])
        hole_idx_extracted = [add_extracted(p) for p in hole_pts_3d]
        index_map = list(loop) + hole_idx_extracted
        extracted_faces_new.extend([index_map[t] for t in tri] for tri in annulus_tris)
        tip_pts = hole_pts_3d + normal * peg_length
        tip_center = centroid + normal * peg_length
        extracted_faces_new.extend(_build_tube_and_cap(hole_idx_extracted, tip_pts, tip_center, add_extracted))

        # remainder side: annulus with a slightly larger hole, plus a recessed socket
        annulus_tris_r = _triangulate_2d(outer_2d, [socket_pts_2d])
        hole_idx_remainder = [add_remainder(p) for p in socket_pts_3d]
        index_map_r = list(loop) + hole_idx_remainder
        remainder_faces_new.extend([index_map_r[t] for t in tri] for tri in annulus_tris_r)
        # bores INTO remainder's own material (remainder's bulk lies on the
        # +normal side, same direction the peg protrudes) -- not outward
        # into the now-empty space on the extracted side, which would add
        # a phantom floating tube instead of a receiving cavity.
        pocket_pts = socket_pts_3d + normal * socket_depth
        pocket_center = centroid + normal * socket_depth
        remainder_faces_new.extend(_build_tube_and_cap(hole_idx_remainder, pocket_pts, pocket_center, add_remainder))

        connector_info = SurfaceCutConnector(
            style=resolved_style,
            radius_mm=peg_radius,
            length_mm=peg_length,
            center_world=centroid,
            axis_world=normal,
        )

    extracted_vertices = (
        np.vstack([mesh.vertices, np.array(extracted_extra)]) if extracted_extra else mesh.vertices.copy()
    )
    remainder_vertices = (
        np.vstack([mesh.vertices, np.array(remainder_extra)]) if remainder_extra else mesh.vertices.copy()
    )
    extracted_faces = np.vstack(
        [mesh.faces[selected_mask], np.array(extracted_faces_new, dtype=np.int64).reshape(-1, 3)]
    )
    remainder_faces = np.vstack(
        [mesh.faces[~selected_mask], np.array(remainder_faces_new, dtype=np.int64).reshape(-1, 3)]
    )

    return SurfaceCutResult(
        extracted_vertices=extracted_vertices,
        extracted_faces=extracted_faces,
        remainder_vertices=remainder_vertices,
        remainder_faces=remainder_faces,
        connector=connector_info,
    )
