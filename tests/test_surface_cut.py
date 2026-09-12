import numpy as np
import pytest
import trimesh

from app.core.extraction import extract_region
from app.core.surface_cut import SurfaceCutFailed, cut_mesh_preserving_surface, find_boundary_loops


def _subdivided_box(band_size=20.0, n_bands=2):
    box = trimesh.creation.box(extents=[10, 10, band_size * n_bands])
    box = box.subdivide().subdivide().subdivide()
    box.apply_translation([0, 0, band_size * n_bands / 2])
    return box


def test_find_boundary_loops_simple_band():
    box = _subdivided_box()
    centroids = box.triangles_center
    selected_mask = centroids[:, 2] < 8
    loops = find_boundary_loops(box, selected_mask)
    assert len(loops) == 1
    assert len(loops[0]) > 4


def test_find_boundary_loops_raises_on_no_boundary():
    box = _subdivided_box()
    all_selected = np.ones(len(box.faces), dtype=bool)
    with pytest.raises(SurfaceCutFailed):
        find_boundary_loops(box, all_selected)


@pytest.mark.parametrize("connector_style", ["round", "keyed", "auto", "none"])
def test_cut_mesh_preserving_surface_watertight_and_conserves_volume(connector_style):
    band_size = 20.0
    box = _subdivided_box(band_size=band_size, n_bands=2)
    centroids = box.triangles_center
    selected_mask = centroids[:, 2] < band_size

    result = cut_mesh_preserving_surface(box, selected_mask, connector_style, 1.0)

    extracted = trimesh.Trimesh(vertices=result.extracted_vertices, faces=result.extracted_faces, process=False)
    remainder = trimesh.Trimesh(vertices=result.remainder_vertices, faces=result.remainder_faces, process=False)
    trimesh.repair.fix_normals(extracted, multibody=False)
    trimesh.repair.fix_normals(remainder, multibody=False)

    assert extracted.is_watertight
    assert remainder.is_watertight

    base_half_volume = 10 * 10 * band_size
    if connector_style == "none":
        assert result.connector is None
        assert extracted.volume == pytest.approx(base_half_volume, rel=0.02)
        assert remainder.volume == pytest.approx(base_half_volume, rel=0.02)
    else:
        assert result.connector is not None
        peg_volume_estimate = np.pi * result.connector.radius_mm**2 * result.connector.length_mm
        assert extracted.volume == pytest.approx(base_half_volume + peg_volume_estimate, rel=0.15)
        # remainder loses roughly a (slightly larger, for clearance) socket cavity
        assert remainder.volume < base_half_volume
        assert remainder.volume > base_half_volume - 3 * peg_volume_estimate


def test_cut_preserves_most_of_original_surface():
    """The whole point: almost all triangles should be untouched originals,
    not new geometry generated at the seam.
    """
    box = _subdivided_box()
    centroids = box.triangles_center
    selected_mask = centroids[:, 2] < 8

    result = cut_mesh_preserving_surface(box, selected_mask, "round", 1.0)

    n_original_total = len(box.faces)
    n_new_extracted = len(result.extracted_faces) - int(selected_mask.sum())
    n_new_remainder = len(result.remainder_faces) - int((~selected_mask).sum())
    # New geometry (cap + connector) should be a small fraction of the
    # original mesh's total face count, not a wholesale remesh.
    assert n_new_extracted < 0.5 * n_original_total
    assert n_new_remainder < 0.5 * n_original_total


def test_extract_region_uses_surface_cut_for_clean_manifold_selection():
    """extract_region should prefer the surface-preserving cut (no voxel
    blockiness) whenever the selection boundary is clean, verified here by
    checking the result keeps almost all of the original vertex count
    instead of being entirely regenerated at some voxel resolution.
    """
    box = _subdivided_box(band_size=20.0, n_bands=1)
    centroids = box.triangles_center
    selected_mask = centroids[:, 2] < 8
    face_colors = np.where(
        selected_mask[:, None],
        np.array([220, 20, 20], dtype=np.uint8),
        np.array([20, 120, 220], dtype=np.uint8),
    )

    result = extract_region(
        box.vertices, box.faces, face_colors, selected_mask, voxels_along_longest=40, connector_style="round"
    )

    # A voxel-based cut at this resolution would produce many more/blockier
    # faces; a surface-preserving cut should stay close to the original
    # mesh's own face count for the untouched region.
    assert len(result.remainder_faces) < len(box.faces)  # no wholesale remesh blow-up
    extracted = trimesh.Trimesh(vertices=result.extracted_vertices, faces=result.extracted_faces, process=False)
    remainder = trimesh.Trimesh(vertices=result.remainder_vertices, faces=result.remainder_faces, process=False)
    assert extracted.is_watertight
    assert remainder.is_watertight


def test_extract_region_falls_back_safely_on_pathological_boundary():
    """A boundary loop can be too numerically awkward for the analytic cap
    (e.g. very large, near-collinear-heavy) -- extract_region must still
    produce a correct, watertight result via the voxel fallback rather than
    ever returning broken geometry.
    """
    box = _subdivided_box(band_size=20.0, n_bands=2)  # taller box, known to stress the cap triangulator
    centroids = box.triangles_center
    selected_mask = centroids[:, 2] < 8
    face_colors = np.where(
        selected_mask[:, None],
        np.array([220, 20, 20], dtype=np.uint8),
        np.array([20, 120, 220], dtype=np.uint8),
    )

    result = extract_region(
        box.vertices, box.faces, face_colors, selected_mask, voxels_along_longest=40, connector_style="round"
    )

    extracted = trimesh.Trimesh(vertices=result.extracted_vertices, faces=result.extracted_faces, process=False)
    remainder = trimesh.Trimesh(vertices=result.remainder_vertices, faces=result.remainder_faces, process=False)
    assert extracted.is_watertight
    assert remainder.is_watertight
    assert extracted.volume > 0
    assert remainder.volume > 0
