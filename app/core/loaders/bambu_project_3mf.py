"""Loader for Bambu Studio / OrcaSlicer / PrusaSlicer "project" 3MF files
that carry per-triangle paint (the result of using the paint/"color
painting" tool, as opposed to assigning a whole object or material a
color). This is the common shape of a "full color" model prepared for
AMS/multi-material printing.

A project 3MF can bundle more than one plate (e.g. a "single color" plate
alongside an "AMS multi color" plate, often duplicating the same geometry
for convenience). We have no reliable, format-defined way to know which
plate the user wants, so we pick the one whose triangles actually use the
most distinct paint colors -- the most informative one for a color split,
and the only sane default when the alternatives are plates that are single
colored anyway.
"""
from __future__ import annotations

import io
import json
import zipfile
from typing import Optional

import numpy as np
from lxml import etree

from app.core.loaders.paint_color import dominant_paint_state
from app.core.model import ColoredMesh

_PROD_PATH_ATTR = "{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}path"


def is_bambu_painted_3mf(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "Metadata/model_settings.config" not in names:
                return False
            model_files = [n for n in names if n.startswith("3D/") and n.endswith(".model")]
            for name in model_files:
                if b"paint_color=" in zf.read(name):
                    return True
    except (zipfile.BadZipFile, KeyError, OSError):
        return False
    return False


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_transform(transform: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    """3MF transform: 12 floats, row-major 3x3 linear part + translation,
    applied to row-vectors as `world = local @ linear + translation`.
    """
    if not transform:
        return np.eye(3), np.zeros(3)
    values = [float(v) for v in transform.split()]
    linear = np.array(values[0:9], dtype=np.float64).reshape(3, 3)
    translation = np.array(values[9:12], dtype=np.float64)
    return linear, translation


def _parse_objects_from_model_xml(xml_bytes: bytes) -> dict[str, dict]:
    """Parse every <object> with an inline <mesh> in one .model XML file.

    Returns {object_id: {"vertices": (N,3) float64, "triangles": [(v1,v2,v3,paint_color|None), ...]}}.
    Objects that only contain <components> (no inline mesh) are omitted.
    """
    objects: dict[str, dict] = {}
    current_id: Optional[str] = None
    current_vertices: list[tuple[float, float, float]] = []
    current_triangles: list[tuple[int, int, int, Optional[str]]] = []

    context = etree.iterparse(io.BytesIO(xml_bytes), events=("start", "end"))
    for event, elem in context:
        tag = _localname(elem.tag)
        if event == "start" and tag == "object":
            current_id = elem.get("id")
            current_vertices = []
            current_triangles = []
        elif event == "end" and tag == "vertex":
            current_vertices.append(
                (float(elem.get("x")), float(elem.get("y")), float(elem.get("z")))
            )
            elem.clear()
        elif event == "end" and tag == "triangle":
            current_triangles.append(
                (
                    int(elem.get("v1")),
                    int(elem.get("v2")),
                    int(elem.get("v3")),
                    elem.get("paint_color"),
                )
            )
            elem.clear()
        elif event == "end" and tag == "object":
            if current_vertices and current_triangles:
                objects[current_id] = {
                    "vertices": np.array(current_vertices, dtype=np.float64),
                    "triangles": current_triangles,
                }
            current_id = None
            elem.clear()

    return objects


def _parse_main_model(xml_bytes: bytes) -> tuple[dict[str, tuple[str, str]], list[tuple[str, str]]]:
    """Parse the top-level 3dmodel.model: which top-level object ids are
    just a reference to a component file (path, inner object id), and the
    build items placing top-level objects into world space.
    """
    root = etree.fromstring(xml_bytes)
    components: dict[str, tuple[str, str]] = {}

    for obj in root.iter():
        if _localname(obj.tag) != "object":
            continue
        oid = obj.get("id")
        for child in obj:
            if _localname(child.tag) != "components":
                continue
            for comp in child:
                if _localname(comp.tag) != "component":
                    continue
                comp_path = comp.get(_PROD_PATH_ATTR) or comp.get("path")
                comp_id = comp.get("objectid")
                if comp_path and comp_id:
                    components[oid] = (comp_path.lstrip("/"), comp_id)
            break

    items: list[tuple[str, str]] = []
    for build in root.iter():
        if _localname(build.tag) != "build":
            continue
        for item in build:
            if _localname(item.tag) != "item":
                continue
            items.append((item.get("objectid"), item.get("transform")))

    return components, items


def _parse_default_extruders(xml_bytes: bytes) -> dict[str, int]:
    """Metadata/model_settings.config: each top-level <object id>'s default
    (unpainted) extruder index.
    """
    root = etree.fromstring(xml_bytes)
    defaults: dict[str, int] = {}
    for obj in root:
        if _localname(obj.tag) != "object":
            continue
        oid = obj.get("id")
        for meta in obj:
            if _localname(meta.tag) == "metadata" and meta.get("key") == "extruder":
                try:
                    defaults[oid] = int(meta.get("value"))
                except (TypeError, ValueError):
                    pass
                break
    return defaults


def _parse_filament_colors(project_settings_bytes: bytes) -> list[tuple[int, int, int]]:
    config = json.loads(project_settings_bytes)
    colors = []
    for hexstr in config.get("filament_colour", []):
        hexstr = hexstr.lstrip("#")
        r = int(hexstr[0:2], 16)
        g = int(hexstr[2:4], 16)
        b = int(hexstr[4:6], 16)
        colors.append((r, g, b))
    return colors


def load_bambu_painted_3mf(path: str) -> ColoredMesh:
    with zipfile.ZipFile(path) as zf:
        main_xml = zf.read("3D/3dmodel.model")
        components, items = _parse_main_model(main_xml)

        default_extruders: dict[str, int] = {}
        if "Metadata/model_settings.config" in zf.namelist():
            default_extruders = _parse_default_extruders(
                zf.read("Metadata/model_settings.config")
            )

        filament_colors: list[tuple[int, int, int]] = []
        if "Metadata/project_settings.config" in zf.namelist():
            filament_colors = _parse_filament_colors(
                zf.read("Metadata/project_settings.config")
            )

        # Resolve each build item's mesh (inline, or via a component file),
        # and score how many distinct paint states it actually uses.
        candidates = []
        model_xml_cache: dict[str, dict[str, dict]] = {"3D/3dmodel.model": None}

        for object_id, transform in items:
            comp = components.get(object_id)
            if comp is not None:
                comp_path, inner_id = comp
                if comp_path not in model_xml_cache:
                    model_xml_cache[comp_path] = _parse_objects_from_model_xml(zf.read(comp_path))
                objects = model_xml_cache[comp_path]
                mesh_data = objects.get(inner_id)
            else:
                if model_xml_cache["3D/3dmodel.model"] is None:
                    model_xml_cache["3D/3dmodel.model"] = _parse_objects_from_model_xml(main_xml)
                mesh_data = model_xml_cache["3D/3dmodel.model"].get(object_id)

            if mesh_data is None:
                continue

            distinct_states = {
                dominant_paint_state(pc) for *_ignore, pc in mesh_data["triangles"] if pc
            }
            candidates.append((object_id, transform, mesh_data, len(distinct_states)))

        if not candidates:
            raise ValueError(f"No paintable mesh objects found in {path!r}")

        object_id, transform, mesh_data, _n_states = max(candidates, key=lambda c: c[3])

        default_extruder = default_extruders.get(object_id, 1)
        default_color = (
            filament_colors[default_extruder - 1]
            if 0 <= default_extruder - 1 < len(filament_colors)
            else (200, 200, 200)
        )

        vertices = mesh_data["vertices"]
        linear, translation = _parse_transform(transform)
        world_vertices = vertices @ linear + translation

        triangles = mesh_data["triangles"]
        faces = np.array([(t[0], t[1], t[2]) for t in triangles], dtype=np.int64)

        face_colors = np.empty((len(triangles), 3), dtype=np.uint8)
        for i, (_v1, _v2, _v3, paint_color) in enumerate(triangles):
            if not paint_color:
                face_colors[i] = default_color
                continue
            state = dominant_paint_state(paint_color)
            if state == 0:
                face_colors[i] = default_color
            elif 0 <= state - 1 < len(filament_colors):
                face_colors[i] = filament_colors[state - 1]
            else:
                face_colors[i] = (200, 200, 200)

        return ColoredMesh(world_vertices, faces, face_colors, source_path=path)
