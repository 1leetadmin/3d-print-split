import numpy as np
import trimesh

from app.core.meshing import weld_local_topology_defects


def _box_with_spurious_sliver_defect(epsilon=1e-4):
    """A watertight box, then deliberately broken the way the real
    duck-model 3MF from Bambu Studio actually was: a spurious extra facet
    reusing one genuine, already-shared edge, but with a near-duplicate
    third corner instead of matching any real triangle. That bumps the
    reused edge to non-manifold (3 faces) and introduces two new tiny
    boundary edges -- exactly the pattern found by inspecting the real
    file's bad edges directly.
    """
    box = trimesh.creation.box(extents=[10, 10, 10])
    vertices = box.vertices.copy()
    faces = box.faces.copy()

    u, v = (int(x) for x in box.edges_sorted[0])
    w = len(vertices)
    vertices = np.vstack([vertices, vertices[u] + epsilon])
    faces = np.vstack([faces, [[u, v, w]]])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh, len(box.faces)  # index of the spurious face


def test_spurious_sliver_defect_is_not_watertight():
    mesh, _ = _box_with_spurious_sliver_defect()
    assert not mesh.is_watertight


def test_weld_local_topology_defects_removes_spurious_sliver():
    mesh, spurious_face_idx = _box_with_spurious_sliver_defect()

    repaired, kept_face_mask = weld_local_topology_defects(mesh)
    trimesh.repair.fix_normals(repaired, multibody=False)

    assert repaired.is_watertight
    assert len(kept_face_mask) == len(mesh.faces)
    assert not kept_face_mask[spurious_face_idx]
    assert kept_face_mask[:spurious_face_idx].all()
    assert len(repaired.faces) == spurious_face_idx


def test_weld_local_topology_defects_leaves_unrelated_geometry_untouched():
    mesh, spurious_face_idx = _box_with_spurious_sliver_defect()
    repaired, kept_face_mask = weld_local_topology_defects(mesh)

    # every real box face (all but the spurious one) survives byte-for-byte,
    # since only the spurious face's own near-duplicate corner is remapped.
    original_box_faces = mesh.faces[:spurious_face_idx]
    assert np.array_equal(repaired.faces, original_box_faces)
    # only the spurious extra vertex should ever be touched; every real
    # box vertex keeps its exact original position.
    assert np.array_equal(repaired.vertices[:-1], mesh.vertices[:-1])


def test_weld_local_topology_defects_is_noop_on_clean_mesh():
    box = trimesh.creation.box(extents=[10, 10, 10])
    repaired, kept_face_mask = weld_local_topology_defects(box)
    assert repaired.is_watertight
    assert kept_face_mask.all()
    assert len(repaired.faces) == len(box.faces)


def test_weld_local_topology_defects_ignores_genuinely_large_gaps():
    """A real (non-tiny) hole must not be welded shut -- that would
    silently distort the model instead of leaving it for hole-filling or
    the voxel fallback to handle correctly.
    """
    box = trimesh.creation.box(extents=[10, 10, 10])
    faces = box.faces
    # drop one face entirely: a real, sizeable hole (edges at full box scale).
    faces_with_hole = np.delete(faces, 0, axis=0)
    holed = trimesh.Trimesh(vertices=box.vertices, faces=faces_with_hole, process=False)
    assert not holed.is_watertight

    repaired, kept_face_mask = weld_local_topology_defects(holed)
    assert not repaired.is_watertight  # left alone, not falsely "fixed"
    assert kept_face_mask.all()
    assert len(repaired.faces) == len(holed.faces)
    assert np.array_equal(repaired.vertices, holed.vertices)


def test_weld_local_topology_defects_bails_out_on_widespread_damage():
    """A mesh shattered into fully disconnected triangles is far past any
    reasonable local-defect size -- it should come back unchanged (still
    non-watertight) rather than the function attempting anything unsafe;
    the caller's voxel fallback is what handles a mesh this broken.
    """
    box = trimesh.creation.box(extents=[10, 10, 10])
    faces = box.faces.copy()
    vertices = box.vertices[faces.flatten()]
    shattered_faces = np.arange(len(vertices)).reshape(-1, 3)
    shattered = trimesh.Trimesh(vertices=vertices, faces=shattered_faces, process=False)
    assert not shattered.is_watertight

    repaired, kept_face_mask = weld_local_topology_defects(shattered, max_bad_vertices=10)
    assert not repaired.is_watertight
    assert kept_face_mask.all()
    assert len(repaired.faces) == len(shattered.faces)
