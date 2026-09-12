"""Interactive click-to-extract: cut the volume under a selected surface
patch away from the rest of a solid, and bake a matching peg/socket
connector into both pieces (directly into the voxel data, before marching
cubes) so they can be printed separately and glued back together.

Doing the connector as voxels rather than a mesh boolean keeps this on the
exact same, already-hardened voxelize -> marching-cubes -> per-component
normal fix pipeline used everywhere else in this app, instead of adding a
second, much less reliable code path (mesh boolean libraries are notorious
for failing on real-world, imperfect input).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import trimesh

from app.core.meshing import transfer_colors_by_nearest_point, voxel_mask_to_watertight_mesh
from app.core.voxelize import choose_pitch

ProgressCB = Optional[Callable[[str, float], None]]

CONNECTOR_STYLES = ("auto", "round", "keyed", "none")


@dataclass
class ConnectorInfo:
    style: str  # resolved style actually used: "round", "keyed", or "none"
    radius_mm: float
    length_mm: float
    center_world: np.ndarray  # seam centroid, world space
    axis_world: np.ndarray  # unit vector, points from extracted part toward remainder


@dataclass
class ExtractionResult:
    extracted_vertices: np.ndarray
    extracted_faces: np.ndarray
    extracted_face_colors: np.ndarray
    remainder_vertices: np.ndarray
    remainder_faces: np.ndarray
    remainder_face_colors: np.ndarray
    connector: Optional[ConnectorInfo]


def _perp_vector(axis: np.ndarray) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(axis, reference)
    return perp / np.linalg.norm(perp)


def _cylinder_mask(
    grid_shape: tuple[int, int, int],
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    axial_min: float,
    axial_max: float,
    flat: bool,
) -> np.ndarray:
    """Boolean mask of a cylinder (or, if ``flat``, a D-shaped cylinder)
    segment along ``axis`` from ``center``, spanning [axial_min, axial_max].
    Only computes within a local bounding box for performance.
    """
    margin = radius + 2.0
    p0 = center + axis * axial_min
    p1 = center + axis * axial_max
    lo = np.floor(np.minimum(p0, p1) - margin).astype(int)
    hi = np.ceil(np.maximum(p0, p1) + margin).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(grid_shape))

    mask = np.zeros(grid_shape, dtype=bool)
    if np.any(hi <= lo):
        return mask

    xs = np.arange(lo[0], hi[0])
    ys = np.arange(lo[1], hi[1])
    zs = np.arange(lo[2], hi[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    rel = np.stack([X - center[0], Y - center[1], Z - center[2]], axis=-1)
    axial = rel @ axis
    radial_vec = rel - axial[..., None] * axis
    radial_dist = np.linalg.norm(radial_vec, axis=-1)
    cond = (axial >= axial_min) & (axial <= axial_max) & (radial_dist <= radius)

    if flat:
        flat_normal = _perp_vector(axis)
        flat_component = radial_vec @ flat_normal
        cond &= flat_component <= radius * 0.4

    mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = cond
    return mask


def _interface_voxels(selected: np.ndarray, remainder: np.ndarray) -> np.ndarray:
    """Selected voxels that are 6-connected-adjacent to a remainder voxel."""
    padded_sel = np.pad(selected, 1, constant_values=False)
    padded_rem = np.pad(remainder, 1, constant_values=False)
    touching = np.zeros_like(padded_sel)
    for shift in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        shifted_rem = np.roll(padded_rem, shift=shift, axis=(0, 1, 2))
        touching |= padded_sel & shifted_rem
    return touching[1:-1, 1:-1, 1:-1]


def _build_connector(
    selected_grid: np.ndarray,
    remainder_grid: np.ndarray,
    style: str,
    scale: float,
) -> Optional[tuple[np.ndarray, np.ndarray, dict]]:
    """Returns (peg_mask, socket_mask, info_dict) or None if no connector
    could be placed (e.g. the two regions don't actually touch).
    """
    interface = _interface_voxels(selected_grid, remainder_grid)
    interface_idx = np.argwhere(interface)
    if len(interface_idx) == 0:
        return None

    selected_centroid = np.argwhere(selected_grid).mean(axis=0)
    remainder_centroid = np.argwhere(remainder_grid).mean(axis=0)
    axis_vec = remainder_centroid - selected_centroid
    norm = np.linalg.norm(axis_vec)
    axis = axis_vec / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

    seam_center = interface_idx.mean(axis=0)
    rel = interface_idx - seam_center
    axial = rel @ axis
    radial_vec = rel - np.outer(axial, axis)
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    transverse_diameter = float(np.percentile(radial_dist, 90) * 2)

    radius = float(np.clip(0.35 * transverse_diameter * scale, 1.5, 12.0))
    length = float(np.clip(2.0 * radius, 2.0, 10.0))
    embed = max(2.0, radius)

    resolved_style = style
    if style == "auto":
        perp1 = _perp_vector(axis)
        perp2 = np.cross(axis, perp1)
        u = radial_vec @ perp1
        v = radial_vec @ perp2
        cov = np.cov(np.stack([u, v]))
        eigvals = np.linalg.eigvalsh(cov)
        aspect = float(np.sqrt(max(eigvals) / max(min(eigvals), 1e-6)))
        resolved_style = "keyed" if (aspect > 1.6 and radius >= 2.5) else "round"

    if resolved_style == "none":
        return None

    flat = resolved_style == "keyed"
    grid_shape = selected_grid.shape
    peg_mask = _cylinder_mask(grid_shape, seam_center, axis, radius, -embed, length, flat)
    clearance = max(1.0, 0.15 * radius)
    socket_mask = _cylinder_mask(
        grid_shape, seam_center, axis, radius + clearance, -0.75, length + 0.75, flat
    )

    info = {
        "style": resolved_style,
        "radius": radius,
        "length": length,
        "center_idx": seam_center,
        "axis": axis,
    }
    return peg_mask, socket_mask, info


def extract_region(
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    body_face_colors: np.ndarray,
    selected_face_mask: np.ndarray,
    voxels_along_longest: int = 120,
    connector_style: str = "auto",
    connector_scale: float = 1.0,
    progress_cb: ProgressCB = None,
) -> ExtractionResult:
    """Cut the selected surface patch's underlying volume away from the rest
    of ``body``, adding a matching peg/socket connector at the seam.
    """
    from scipy.spatial import cKDTree  # local import: heavy, only needed here

    def report(msg: str, frac: float) -> None:
        if progress_cb is not None:
            progress_cb(msg, frac)

    if connector_style not in CONNECTOR_STYLES:
        raise ValueError(f"Unknown connector_style {connector_style!r}, expected one of {CONNECTOR_STYLES}")
    if not np.any(selected_face_mask):
        raise ValueError("Selection is empty -- nothing to extract")
    if np.all(selected_face_mask):
        raise ValueError("Selection covers the entire model -- nothing would be left")

    report("Building solid", 0.0)
    body_mesh = trimesh.Trimesh(
        vertices=body_vertices, faces=body_faces, face_colors=body_face_colors, process=False
    )
    pitch = choose_pitch(body_vertices, voxels_along_longest)

    report("Voxelizing", 0.1)
    voxel = body_mesh.voxelized(pitch=pitch).fill()
    dense = np.asarray(voxel.matrix, dtype=bool)
    transform = np.asarray(voxel.transform, dtype=np.float64)

    n_samples = int(np.clip(len(body_mesh.faces) * 4, 2_000, 400_000))
    report(f"Sampling {n_samples} surface points", 0.25)
    sample_points, sample_face_ids = trimesh.sample.sample_surface(body_mesh, n_samples)
    tree = cKDTree(sample_points)

    voxel_indices = np.argwhere(dense)
    if len(voxel_indices) == 0:
        raise ValueError("Voxelization produced no solid voxels; try a finer resolution")
    homogeneous = np.hstack([voxel_indices.astype(np.float64), np.ones((len(voxel_indices), 1))])
    world_points = (transform @ homogeneous.T).T[:, :3]

    report("Labeling voxels selected vs. remainder", 0.4)
    _dist, nearest_sample_idx = tree.query(world_points, k=1)
    voxel_is_selected = selected_face_mask[sample_face_ids[nearest_sample_idx]]

    selected_grid = np.zeros_like(dense)
    remainder_grid = np.zeros_like(dense)
    sel_idx = voxel_indices[voxel_is_selected]
    rem_idx = voxel_indices[~voxel_is_selected]
    selected_grid[sel_idx[:, 0], sel_idx[:, 1], sel_idx[:, 2]] = True
    remainder_grid[rem_idx[:, 0], rem_idx[:, 1], rem_idx[:, 2]] = True

    if not selected_grid.any():
        raise ValueError("Selection didn't cover any solid volume -- try a lower click tolerance")
    if not remainder_grid.any():
        raise ValueError("Selection covers the entire solid volume -- nothing would be left")

    connector_info = None
    if connector_style != "none":
        report("Building connector", 0.6)
        built = _build_connector(selected_grid, remainder_grid, connector_style, connector_scale)
        if built is not None:
            peg_mask, socket_mask, info = built
            selected_grid = selected_grid | peg_mask
            remainder_grid = remainder_grid & ~socket_mask
            center_world = (transform @ np.append(info["center_idx"], 1.0))[:3]
            axis_world = transform[:3, :3] @ info["axis"]
            axis_world = axis_world / np.linalg.norm(axis_world)
            connector_info = ConnectorInfo(
                style=info["style"],
                radius_mm=info["radius"] * pitch,
                length_mm=info["length"] * pitch,
                center_world=center_world,
                axis_world=axis_world,
            )

    report("Reconstructing extracted part", 0.75)
    extracted_vertices, extracted_faces = voxel_mask_to_watertight_mesh(selected_grid, transform)
    report("Reconstructing remainder", 0.85)
    remainder_vertices, remainder_faces = voxel_mask_to_watertight_mesh(remainder_grid, transform)

    report("Coloring parts", 0.95)
    extracted_colors = transfer_colors_by_nearest_point(
        extracted_vertices, extracted_faces, tree, sample_face_ids, body_face_colors
    )
    remainder_colors = transfer_colors_by_nearest_point(
        remainder_vertices, remainder_faces, tree, sample_face_ids, body_face_colors
    )

    report("Done", 1.0)
    return ExtractionResult(
        extracted_vertices=extracted_vertices,
        extracted_faces=extracted_faces,
        extracted_face_colors=extracted_colors,
        remainder_vertices=remainder_vertices,
        remainder_faces=remainder_faces,
        remainder_face_colors=remainder_colors,
        connector=connector_info,
    )
