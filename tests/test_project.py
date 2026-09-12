import numpy as np
import trimesh

from app.core.model import ColoredMesh
from app.core.project import Project


def _duck_like_model():
    """A body with two embedded appendages of different colors, e.g. a
    body + beak + antenna, so we can do two sequential extractions.
    """
    body = trimesh.creation.box(extents=[20, 20, 20])
    beak = trimesh.creation.box(extents=[8, 6, 6])
    beak.apply_translation([12, 0, 0])
    antenna = trimesh.creation.box(extents=[4, 4, 8])
    antenna.apply_translation([0, 0, 12])

    red = np.tile(np.array([220, 20, 20], dtype=np.uint8), (len(body.faces), 1))
    blue = np.tile(np.array([20, 120, 220], dtype=np.uint8), (len(beak.faces), 1))
    green = np.tile(np.array([20, 200, 20], dtype=np.uint8), (len(antenna.faces), 1))

    vertices = np.vstack([body.vertices, beak.vertices, antenna.vertices])
    faces = np.vstack(
        [body.faces, beak.faces + len(body.vertices), antenna.faces + len(body.vertices) + len(beak.vertices)]
    )
    face_colors = np.vstack([red, blue, green])
    return ColoredMesh(vertices, faces, face_colors)


def _first_face_with_color(body, color):
    matches = np.where(np.all(body.face_colors == np.array(color), axis=1))[0]
    assert len(matches) > 0, f"no face with color {color} found"
    return int(matches[0])


def test_incremental_mode_shrinks_body_across_extractions():
    project = Project(_duck_like_model(), voxels_along_longest=50, independent_mode=False)
    body_before = project.body

    beak_face = _first_face_with_color(project.body, [20, 120, 220])
    mask = project.select(beak_face, tolerance=10)
    project.extract(mask, name="beak", connector_style="round")

    assert len(project.parts) == 1
    assert project.body is not body_before  # incremental mode: body shrank

    antenna_face = _first_face_with_color(project.body, [20, 200, 20])
    mask2 = project.select(antenna_face, tolerance=10)
    project.extract(mask2, name="antenna", connector_style="round")

    assert len(project.parts) == 2
    assert project.parts[0].name == "beak"
    assert project.parts[1].name == "antenna"


def test_independent_mode_keeps_body_unchanged():
    project = Project(_duck_like_model(), voxels_along_longest=50, independent_mode=True)
    body_before = project.body

    beak_face = _first_face_with_color(project.body, [20, 120, 220])
    mask = project.select(beak_face, tolerance=10)
    project.extract(mask, name="beak", connector_style="round")

    assert len(project.parts) == 1
    assert project.body is body_before  # independent mode: body never changes


def test_undo_redo_restores_state():
    project = Project(_duck_like_model(), voxels_along_longest=50, independent_mode=False)
    original_body = project.body
    assert not project.can_undo
    assert not project.can_redo

    beak_face = _first_face_with_color(project.body, [20, 120, 220])
    mask = project.select(beak_face, tolerance=10)
    project.extract(mask, name="beak", connector_style="round")

    body_after_extract = project.body
    assert len(project.parts) == 1
    assert project.can_undo
    assert not project.can_redo

    project.undo()
    assert project.body is original_body
    assert len(project.parts) == 0
    assert not project.can_undo
    assert project.can_redo

    project.redo()
    assert project.body is body_after_extract
    assert len(project.parts) == 1
    assert project.can_undo
    assert not project.can_redo


def test_export_parts_writes_manifest_and_stls(tmp_path):
    project = Project(_duck_like_model(), voxels_along_longest=50, independent_mode=False)
    beak_face = _first_face_with_color(project.body, [20, 120, 220])
    mask = project.select(beak_face, tolerance=10)
    project.extract(mask, name="beak", connector_style="round")

    manifest_path = project.export_parts(str(tmp_path))
    import json

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert len(manifest["parts"]) == 2  # beak + remaining_body
    names = {p["name"] for p in manifest["parts"]}
    assert names == {"beak", "remaining_body"}
    for p in manifest["parts"]:
        assert p["is_watertight"] is True
        assert (tmp_path / p["filename"]).exists()
