"""Loader for the non-standard binary-STL color extension used by tools such
as older Cura, Simplify3D and Materialise Magics.

Binary STL reserves a 2-byte "attribute byte count" per triangle that these
tools repurpose to store a 15-bit color: bit 15 marks the color as valid for
that triangle, and bits 14-10 / 9-5 / 4-0 give 5-bit red / green / blue.
Materialise's convention additionally allows an 80-byte-header default color
("COLOR=") used for any triangle whose valid bit is unset.

Plain ASCII STL and binary STL without this extension have no color data at
all; such files load as a single default-colored part (nothing to split).
"""
from __future__ import annotations

import struct

import numpy as np
from stl import mesh as stl_mesh

from app.core.model import ColoredMesh

_DEFAULT_COLOR = np.array([200, 200, 200], dtype=np.uint8)
_VALID_BIT = 0x8000


def _bits_to_rgb(attr: np.ndarray, swap_rb: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Decode RGB555 attribute values. Returns (rgb, valid_mask)."""
    valid = (attr & _VALID_BIT) != 0
    r5 = (attr >> 10) & 0x1F
    g5 = (attr >> 5) & 0x1F
    b5 = attr & 0x1F
    # Expand 5-bit channel to 8-bit by replicating the top bits.
    r8 = ((r5 << 3) | (r5 >> 2)).astype(np.uint8)
    g8 = ((g5 << 3) | (g5 >> 2)).astype(np.uint8)
    b8 = ((b5 << 3) | (b5 >> 2)).astype(np.uint8)
    if swap_rb:
        r8, b8 = b8, r8
    rgb = np.stack([r8, g8, b8], axis=1)
    return rgb, valid


def _read_header_default_color(path: str) -> np.ndarray | None:
    """Parse a Magics-style 'COLOR=' marker from the 80-byte binary STL header."""
    with open(path, "rb") as f:
        header = f.read(80)
    marker = b"COLOR="
    idx = header.find(marker)
    if idx == -1 or idx + len(marker) + 4 > len(header):
        return None
    r, g, b, _a = struct.unpack_from("BBBB", header, idx + len(marker))
    return np.array([r, g, b], dtype=np.uint8)


def is_binary_stl(path: str) -> bool:
    with open(path, "rb") as f:
        header = f.read(5)
    if header[:5].lower() == b"solid":
        # Could still be binary if it happens to start with "solid"; fall
        # back to a size check.
        import os

        size = os.path.getsize(path)
        if size < 84:
            return False
        with open(path, "rb") as f2:
            f2.seek(80)
            (tri_count,) = struct.unpack("<I", f2.read(4))
        expected = 84 + tri_count * 50
        return size == expected
    return True


def load_stl_with_color(path: str, swap_rb: bool = False) -> ColoredMesh:
    m = stl_mesh.Mesh.from_file(path)
    vertices = m.vectors.reshape(-1, 3)  # (3*M, 3), not yet deduplicated
    faces = np.arange(len(vertices)).reshape(-1, 3)

    attr = np.asarray(m.attr, dtype=np.uint16).reshape(-1)
    rgb, valid = _bits_to_rgb(attr, swap_rb=swap_rb)

    default = _read_header_default_color(path)
    if default is None:
        default = _DEFAULT_COLOR

    face_colors = np.where(valid[:, None], rgb, default)

    return ColoredMesh(vertices, faces, face_colors.astype(np.uint8), source_path=path)
