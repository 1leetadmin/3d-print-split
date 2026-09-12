"""Pick the right loader for an input file and return a ColoredMesh."""
from __future__ import annotations

import os

from app.core.model import ColoredMesh
from app.core.loaders.bambu_project_3mf import is_bambu_painted_3mf, load_bambu_painted_3mf
from app.core.loaders.generic_trimesh import load_via_trimesh
from app.core.loaders.stl_color import is_binary_stl, load_stl_with_color

SUPPORTED_EXTENSIONS = {".3mf", ".obj", ".ply", ".stl"}


def load_colored_mesh(path: str, swap_rb: bool = False) -> ColoredMesh:
    """Load any supported file into a :class:`ColoredMesh`.

    ``swap_rb`` only affects binary STL color decoding, where the 5-5-5 bit
    order is an unofficial convention that a few tools implement backwards --
    flip it if a loaded model's colors look like red/blue are swapped.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".stl":
        if is_binary_stl(path):
            return load_stl_with_color(path, swap_rb=swap_rb)
        # ASCII STL never carries color; still load it as a (single-color) mesh.
        return load_via_trimesh(path)

    if ext == ".3mf" and is_bambu_painted_3mf(path):
        # A project 3MF with paint-tool color data: trimesh's generic 3MF
        # loader doesn't know about this vendor extension and would see no
        # color at all, so handle it before falling through to that path.
        return load_bambu_painted_3mf(path)

    return load_via_trimesh(path)
