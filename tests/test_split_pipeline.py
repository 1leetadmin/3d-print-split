import numpy as np
import pytest
import trimesh

from app.core.model import ColoredMesh
from app.core.split import _part_for_label, split_by_color
from app.core.voxelize import VoxelLabeling


def _colored_stack(n_bands: int, band_size: float = 20.0) -> ColoredMesh:
    """A tall box made of ``n_bands`` stacked horizontal color bands, each its
    own distinct color, sharing one continuous mesh (as if a slicer had
    painted each band a different color).
    """
    total_height = band_size * n_bands
    box = trimesh.creation.box(extents=[band_size, band_size, total_height])
    box.apply_translation([0, 0, total_height / 2])  # z in [0, total_height]

    palette = np.array(
        [[220, 20, 20], [20, 120, 220], [20, 200, 20], [230, 200, 20]], dtype=np.uint8
    )
    centroids = box.triangles_center
    band_idx = np.clip((centroids[:, 2] // band_size).astype(int), 0, n_bands - 1)
    face_colors = palette[band_idx % len(palette)]

    return ColoredMesh(vertices=box.vertices, faces=box.faces, face_colors=face_colors)


@pytest.mark.parametrize("n_bands", [2, 3])
def test_split_produces_one_watertight_part_per_color(n_bands):
    band_size = 20.0
    colored_mesh = _colored_stack(n_bands, band_size=band_size)

    parts = split_by_color(colored_mesh, voxels_along_longest=60 * n_bands, max_colors=8)

    assert len(parts) == n_bands

    total_volume = 0.0
    seen_colors = set()
    for part in parts:
        mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
        assert mesh.is_watertight, f"part {part.hex_color} is not watertight"
        assert mesh.volume > 0
        total_volume += mesh.volume
        seen_colors.add(part.color_rgb)

    assert len(seen_colors) == n_bands  # every part got a distinct color

    expected_volume = band_size * band_size * band_size * n_bands
    # Voxelization + marching cubes at finite pitch pads the solid by
    # roughly one voxel shell, so volume converges to the true value only in
    # the limit of finer resolution (verified separately); at this test's
    # resolution the bias is a few percent.
    assert total_volume == pytest.approx(expected_volume, rel=0.12)


def test_split_orders_bands_along_original_axis():
    colored_mesh = _colored_stack(2, band_size=20.0)
    parts = split_by_color(colored_mesh, voxels_along_longest=30, max_colors=8)
    assert len(parts) == 2

    by_color = {p.color_rgb: p for p in parts}
    red = by_color[(220, 20, 20)]
    blue = by_color[(20, 120, 220)]

    red_mesh = trimesh.Trimesh(vertices=red.vertices, faces=red.faces, process=False)
    blue_mesh = trimesh.Trimesh(vertices=blue.vertices, faces=blue.faces, process=False)

    # Red band was painted for z in [0, 20], blue for z in [20, 40]: the
    # reconstructed solids should preserve that ordering along z.
    assert red_mesh.bounds[1][2] <= blue_mesh.bounds[1][2]
    assert red_mesh.centroid[2] < blue_mesh.centroid[2]


def test_touching_objects_each_keep_full_volume():
    """Two boxes glued exactly face-to-face, as a multi-object 3MF/OBJ that's
    already split one part per color would produce. Merging them into one
    mesh before voxelizing creates a non-manifold seam and can silently
    drop most of a part's volume -- component_ids keeps them independent.
    """
    size = 10.0
    box1 = trimesh.creation.box(extents=[size, size, size])
    box1.apply_translation([0, 0, size / 2])
    box2 = trimesh.creation.box(extents=[size, size, size])
    box2.apply_translation([0, 0, size * 1.5])  # sits exactly on top of box1

    vertices = np.vstack([box1.vertices, box2.vertices])
    faces = np.vstack([box1.faces, box2.faces + len(box1.vertices)])
    face_colors = np.vstack(
        [
            np.tile(np.array([220, 20, 20], dtype=np.uint8), (len(box1.faces), 1)),
            np.tile(np.array([20, 120, 220], dtype=np.uint8), (len(box2.faces), 1)),
        ]
    )
    component_ids = np.concatenate(
        [np.zeros(len(box1.faces), dtype=np.int64), np.ones(len(box2.faces), dtype=np.int64)]
    )
    colored_mesh = ColoredMesh(vertices, faces, face_colors, component_ids=component_ids)

    parts = split_by_color(colored_mesh, voxels_along_longest=120, max_colors=8)

    assert len(parts) == 2
    expected_volume = size**3
    for part in parts:
        mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
        assert mesh.is_watertight
        assert mesh.volume == pytest.approx(expected_volume, rel=0.12)


def test_ambiguous_diagonal_touch_stays_watertight():
    """Two solid voxels touching only at a shared corner is the classic
    marching-cubes topological-ambiguity case: it can leave a non-manifold
    edge (shared by 3+ faces) where the two "islands" meet, even though the
    surface has no open boundary. Real color regions on complex models are
    frequently fragmented into many islands, and any two of them can end up
    exactly diagonally adjacent like this by chance.
    """
    palette = np.array([[200, 100, 50]], dtype=np.uint8)
    labeling = VoxelLabeling(
        palette=palette,
        voxel_indices=np.array([[0, 0, 0], [1, 1, 1]]),
        voxel_labels=np.array([0, 0]),
        transform=np.eye(4),
        grid_shape=(2, 2, 2),
        pitch=1.0,
    )

    part = _part_for_label(0, palette, labeling)

    assert part is not None
    mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_single_color_mesh_yields_one_part():
    box = trimesh.creation.box(extents=[10, 10, 10])
    face_colors = np.tile(np.array([100, 100, 100], dtype=np.uint8), (len(box.faces), 1))
    colored_mesh = ColoredMesh(vertices=box.vertices, faces=box.faces, face_colors=face_colors)

    parts = split_by_color(colored_mesh, voxels_along_longest=20, max_colors=8)

    assert len(parts) == 1
    assert parts[0].color_rgb == (100, 100, 100)
