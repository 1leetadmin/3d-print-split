"""Write each color's solid as its own printable STL, plus a manifest and a
combined colored preview so you can sanity-check that the parts still line
up before you print and glue them.
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh

from app.core.model import ColorPart


def export_parts(parts: list[ColorPart], out_dir: str) -> str:
    """Export STL per part + manifest.json + assembly_preview.ply.

    Returns the manifest path.
    """
    os.makedirs(out_dir, exist_ok=True)

    manifest = {"parts": []}
    preview_vertices = []
    preview_colors = []
    preview_faces = []
    vertex_offset = 0

    for i, part in enumerate(parts):
        filename = f"part_{i + 1:02d}_{part.hex_color}.stl"
        path = os.path.join(out_dir, filename)

        mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
        mesh.export(path, file_type="stl")

        mins, maxs = mesh.bounds
        manifest["parts"].append(
            {
                "filename": filename,
                "color_rgb": list(part.color_rgb),
                "color_hex": f"#{part.hex_color}",
                "voxel_count": part.voxel_count,
                "is_watertight": bool(mesh.is_watertight),
                "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
                "bounds_min": mins.tolist(),
                "bounds_max": maxs.tolist(),
            }
        )

        preview_vertices.append(mesh.vertices)
        preview_faces.append(mesh.faces + vertex_offset)
        preview_colors.append(np.tile(np.array(part.color_rgb, dtype=np.uint8), (len(mesh.vertices), 1)))
        vertex_offset += len(mesh.vertices)

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if preview_vertices:
        combined = trimesh.Trimesh(
            vertices=np.vstack(preview_vertices),
            faces=np.vstack(preview_faces),
            vertex_colors=np.vstack(preview_colors),
            process=False,
        )
        combined.export(os.path.join(out_dir, "assembly_preview.ply"), file_type="ply")

    return manifest_path
