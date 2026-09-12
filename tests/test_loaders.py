import struct

import numpy as np
import trimesh

from app.core.loaders.detect import load_colored_mesh
from app.core.loaders.stl_color import _bits_to_rgb


def _write_color_binary_stl(path, triangles, attrs, header=b""):
    """Minimal binary STL writer for building test fixtures.

    ``triangles`` is a list of (v1, v2, v3) each a 3-tuple xyz.
    ``attrs`` is a parallel list of uint16 attribute-byte-count values.
    """
    with open(path, "wb") as f:
        header_bytes = header.ljust(80, b"\x00")[:80]
        f.write(header_bytes)
        f.write(struct.pack("<I", len(triangles)))
        for (v1, v2, v3), attr in zip(triangles, attrs):
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))  # normal (unused by our loader)
            for v in (v1, v2, v3):
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", attr))


def test_stl_color_attribute_decoding_direct():
    # bit15 valid, R=31 (5 bits, top of range), G=0, B=0 -> pure red (255,0,0) after expansion
    attr_red = 0x8000 | (0x1F << 10)
    rgb, valid = _bits_to_rgb(np.array([attr_red], dtype=np.uint16))
    assert valid[0]
    assert tuple(rgb[0]) == (255, 0, 0)

    # valid bit unset -> caller should fall back to default color, not this decode
    rgb2, valid2 = _bits_to_rgb(np.array([0x0000], dtype=np.uint16))
    assert not valid2[0]


def test_load_color_stl_end_to_end(tmp_path):
    path = tmp_path / "two_tri.stl"
    red_attr = 0x8000 | (0x1F << 10)  # pure red, valid
    blue_attr = 0x8000 | 0x1F  # pure blue, valid
    triangles = [
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 1), (1, 0, 1), (0, 1, 1)),
    ]
    _write_color_binary_stl(path, triangles, [red_attr, blue_attr])

    mesh = load_colored_mesh(str(path))

    assert mesh.faces.shape == (2, 3)
    colors = {tuple(c) for c in mesh.face_colors}
    assert (255, 0, 0) in colors
    assert (0, 0, 255) in colors


def test_load_color_stl_falls_back_to_header_default(tmp_path):
    path = tmp_path / "default_color.stl"
    header = b"COLOR=" + bytes([10, 200, 30, 255])
    unset_attr = 0x0000  # valid bit not set -> use header default
    triangles = [((0, 0, 0), (1, 0, 0), (0, 1, 0))]
    _write_color_binary_stl(path, triangles, [unset_attr], header=header)

    mesh = load_colored_mesh(str(path))

    assert tuple(mesh.face_colors[0]) == (10, 200, 30)


def test_load_vertex_colored_ply_two_parts(tmp_path):
    box1 = trimesh.creation.box(extents=[10, 10, 10])
    box1.apply_translation([0, 0, 5])
    box2 = trimesh.creation.box(extents=[10, 10, 10])
    box2.apply_translation([0, 0, 15])

    red = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), (len(box1.vertices), 1))
    blue = np.tile(np.array([0, 0, 255, 255], dtype=np.uint8), (len(box2.vertices), 1))

    vertices = np.vstack([box1.vertices, box2.vertices])
    faces = np.vstack([box1.faces, box2.faces + len(box1.vertices)])
    vertex_colors = np.vstack([red, blue])

    combined = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    path = tmp_path / "two_box.ply"
    combined.export(str(path), file_type="ply")

    mesh = load_colored_mesh(str(path))
    colors = {tuple(c) for c in mesh.face_colors}
    assert (255, 0, 0) in colors
    assert (0, 0, 255) in colors

    # Faces belonging to box1 (all vertices z==5 side) should all be red,
    # and box2's should all be blue -- no bleed since the boxes don't share vertices.
    box1_face_colors = mesh.face_colors[: len(box1.faces)]
    box2_face_colors = mesh.face_colors[len(box1.faces) :]
    assert np.all(box1_face_colors == np.array([255, 0, 0]))
    assert np.all(box2_face_colors == np.array([0, 0, 255]))
