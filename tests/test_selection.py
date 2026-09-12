import numpy as np
import trimesh

from app.core.selection import build_face_adjacency_list, flood_fill_by_color


def _two_color_box():
    """A box with the top half painted a different color than the bottom
    half, sharing one continuous mesh -- subdivided so each side has more
    than one face (a bare box's sides span the full height in one face).
    """
    box = trimesh.creation.box(extents=[10, 10, 20])
    box = box.subdivide().subdivide()
    box.apply_translation([0, 0, 10])  # z in [0, 20]

    centroids = box.triangles_center
    face_colors = np.where(
        (centroids[:, 2] < 10)[:, None],
        np.array([220, 20, 20], dtype=np.uint8),
        np.array([20, 120, 220], dtype=np.uint8),
    )
    box.visual.face_colors = np.hstack(
        [face_colors, np.full((len(face_colors), 1), 255, dtype=np.uint8)]
    )
    return box, face_colors


def test_flood_fill_selects_only_matching_contiguous_region():
    box, face_colors = _two_color_box()
    adjacency_list = build_face_adjacency_list(box)

    red_seed = int(np.where((face_colors == [220, 20, 20]).all(axis=1))[0][0])
    selected = flood_fill_by_color(face_colors, adjacency_list, red_seed, tolerance=10)

    assert selected.sum() > 0
    assert np.all(face_colors[selected] == [220, 20, 20])
    # None of the blue faces should have been picked up.
    blue_mask = (face_colors == [20, 120, 220]).all(axis=1)
    assert not np.any(selected & blue_mask)


def test_flood_fill_high_tolerance_reaches_whole_connected_mesh():
    box, face_colors = _two_color_box()
    adjacency_list = build_face_adjacency_list(box)

    red_seed = int(np.where((face_colors == [220, 20, 20]).all(axis=1))[0][0])
    selected = flood_fill_by_color(face_colors, adjacency_list, red_seed, tolerance=1000)

    # A tolerance wider than the whole color cube should select every face
    # connected to the seed -- here, the entire (single connected) mesh.
    assert selected.sum() == len(face_colors)


def test_flood_fill_seed_always_included():
    box, face_colors = _two_color_box()
    adjacency_list = build_face_adjacency_list(box)
    selected = flood_fill_by_color(face_colors, adjacency_list, 0, tolerance=0)
    assert selected[0]
