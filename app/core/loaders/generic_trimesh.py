"""Loader for formats trimesh already understands well: 3MF, OBJ(+MTL), PLY.

trimesh's 3MF importer follows the Materials/Production extensions (basematerials
+ per-triangle pid/p1, or per-object materials), its OBJ importer follows
mtllib/usemtl Kd colors as well as the common "v x y z r g b" vertex-color
extension, and its PLY importer follows per-vertex color. This module turns
whatever trimesh gives back (a single Trimesh or a multi-geometry Scene) into
one flat :class:`ColoredMesh`.

Known limitation: slicer-proprietary sub-triangle paint formats (e.g.
PrusaSlicer/Bambu Studio "mmu_segmentation" multi-material painting on a
single mesh) are not decoded. If a 3MF only carries paint data in that form,
every triangle will come back as a single default color -- re-export the
model as separate colored objects/materials, or as a vertex-colored OBJ/PLY,
instead.
"""
from __future__ import annotations

from itertools import cycle

import numpy as np
import trimesh

from app.core.model import ColoredMesh

# trimesh fills in this exact gray whenever a mesh has no assigned color at
# all (trimesh.visual.color.DEFAULT_COLOR) -- used below to detect "no real
# color/material data anywhere in this file" rather than an intentional gray.
_DEFAULT_COLOR = np.array([102, 102, 102], dtype=np.uint8)

# Distinct, readable fallback palette used only when a multi-object file
# carries no color/material information at all -- each object still needs to
# become its own printable part.
_SYNTHETIC_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0),
]


def face_colors_from_trimesh(mesh: trimesh.Trimesh) -> np.ndarray:
    """Best-effort per-face RGB for any trimesh visual type."""
    visual = mesh.visual
    if hasattr(visual, "to_color"):
        try:
            visual = visual.to_color()
        except Exception:
            pass
    face_colors = getattr(visual, "face_colors", None)
    if face_colors is not None and len(face_colors) == len(mesh.faces):
        return np.asarray(face_colors)[:, :3].astype(np.uint8)
    return np.tile(_DEFAULT_COLOR, (len(mesh.faces), 1))


def _iter_scene_instances(scene: trimesh.Scene):
    """Yield (geometry, world_transform) for every placed instance, so a
    geometry referenced by more than one build item is duplicated correctly.
    """
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph.get(node_name)
        geom = scene.geometry.get(geom_name)
        if geom is not None and len(geom.faces) > 0:
            yield geom, transform


def load_via_trimesh(path: str) -> ColoredMesh:
    loaded = trimesh.load(path, process=False, force=None)

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_face_colors: list[np.ndarray] = []
    all_component_ids: list[np.ndarray] = []
    vertex_offset = 0
    geometry_count = 0

    if isinstance(loaded, trimesh.Scene):
        for geom, transform in _iter_scene_instances(loaded):
            component_id = geometry_count
            geometry_count += 1
            verts = trimesh.transformations.transform_points(geom.vertices, transform)
            fc = face_colors_from_trimesh(geom)
            all_vertices.append(verts)
            all_faces.append(geom.faces + vertex_offset)
            all_face_colors.append(fc)
            all_component_ids.append(np.full(len(geom.faces), component_id, dtype=np.int64))
            vertex_offset += len(verts)
    else:
        mesh = loaded
        geometry_count = 1
        all_vertices.append(np.asarray(mesh.vertices))
        all_faces.append(np.asarray(mesh.faces))
        all_face_colors.append(face_colors_from_trimesh(mesh))
        all_component_ids.append(np.zeros(len(mesh.faces), dtype=np.int64))

    if not all_vertices:
        raise ValueError(f"No mesh geometry found in {path!r}")

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    face_colors = np.vstack(all_face_colors)
    # Multiple original objects are kept distinct through the whole pipeline
    # (never merged into one mesh) so two objects placed touching each other
    # -- the normal shape of a file that's already split one part per color
    # -- don't create a non-manifold seam. A single-geometry file has no
    # such natural split, so component_ids is left as None.
    component_ids = np.concatenate(all_component_ids) if geometry_count > 1 else None

    # If nothing carried real color/material info (every face fell back to
    # the default gray) but the file is split into multiple objects, treat
    # "one object == one part" and hand each object a distinct color instead
    # of silently collapsing everything into a single uncuttable part.
    all_default = bool(np.all(face_colors == _DEFAULT_COLOR))
    if all_default and geometry_count > 1:
        palette = cycle(_SYNTHETIC_PALETTE)
        recolored = []
        for fc in all_face_colors:
            color = np.array(next(palette), dtype=np.uint8)
            recolored.append(np.tile(color, (len(fc), 1)))
        face_colors = np.vstack(recolored)

    return ColoredMesh(vertices, faces, face_colors, source_path=path, component_ids=component_ids)
