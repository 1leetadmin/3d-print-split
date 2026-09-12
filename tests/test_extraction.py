import numpy as np
import pytest
import trimesh

from app.core.extraction import extract_region


def _body_with_appendage():
    """A big box with a smaller box ("beak") embedded into one of its faces
    -- overlapping, self-intersecting where they meet, exactly like a real
    painted model's mesh would be at a color boundary. Voxelization treats
    this as one solid union regardless of the raw triangle soup's quality.
    """
    body = trimesh.creation.box(extents=[20, 20, 20])
    beak = trimesh.creation.box(extents=[8, 6, 6])
    beak.apply_translation([12, 0, 0])  # body spans x in [-10,10]; beak x in [8,16]

    body_colors = np.tile(np.array([220, 20, 20], dtype=np.uint8), (len(body.faces), 1))
    beak_colors = np.tile(np.array([20, 120, 220], dtype=np.uint8), (len(beak.faces), 1))

    vertices = np.vstack([body.vertices, beak.vertices])
    faces = np.vstack([body.faces, beak.faces + len(body.vertices)])
    face_colors = np.vstack([body_colors, beak_colors])
    beak_face_mask = np.zeros(len(faces), dtype=bool)
    beak_face_mask[len(body.faces) :] = True
    return vertices, faces, face_colors, beak_face_mask


@pytest.mark.parametrize("connector_style", ["round", "keyed", "auto"])
def test_extract_region_produces_two_watertight_parts_with_connector(connector_style):
    vertices, faces, face_colors, beak_mask = _body_with_appendage()

    result = extract_region(
        vertices,
        faces,
        face_colors,
        beak_mask,
        voxels_along_longest=60,
        connector_style=connector_style,
    )

    extracted = trimesh.Trimesh(
        vertices=result.extracted_vertices, faces=result.extracted_faces, process=False
    )
    remainder = trimesh.Trimesh(
        vertices=result.remainder_vertices, faces=result.remainder_faces, process=False
    )

    assert extracted.is_watertight
    assert remainder.is_watertight
    assert extracted.volume > 0
    assert remainder.volume > 0

    assert result.connector is not None
    assert result.connector.style in ("round", "keyed")
    assert result.connector.radius_mm > 0
    assert result.connector.length_mm > 0

    # The extracted part (beak ~8x6x6 plus a peg) should be much smaller
    # than the remainder (body ~20x20x20 minus a small socket cavity).
    assert extracted.volume < remainder.volume


def test_extract_region_connector_none_skips_connector():
    vertices, faces, face_colors, beak_mask = _body_with_appendage()

    result = extract_region(
        vertices, faces, face_colors, beak_mask, voxels_along_longest=60, connector_style="none"
    )

    assert result.connector is None
    extracted = trimesh.Trimesh(
        vertices=result.extracted_vertices, faces=result.extracted_faces, process=False
    )
    remainder = trimesh.Trimesh(
        vertices=result.remainder_vertices, faces=result.remainder_faces, process=False
    )
    assert extracted.is_watertight
    assert remainder.is_watertight


def test_extract_region_rejects_empty_or_full_selection():
    vertices, faces, face_colors, beak_mask = _body_with_appendage()

    with pytest.raises(ValueError):
        extract_region(vertices, faces, face_colors, np.zeros(len(faces), dtype=bool))

    with pytest.raises(ValueError):
        extract_region(vertices, faces, face_colors, np.ones(len(faces), dtype=bool))


def test_extracted_and_remainder_colors_roughly_preserved():
    vertices, faces, face_colors, beak_mask = _body_with_appendage()

    result = extract_region(
        vertices, faces, face_colors, beak_mask, voxels_along_longest=60, connector_style="round"
    )

    # Most of the extracted part's faces should be near the beak's blue,
    # and most of the remainder's faces near the body's red (a few near the
    # cut/connector may pick up the other color, which is fine).
    blue = np.array([20, 120, 220])
    red = np.array([220, 20, 20])
    extracted_blue_frac = np.mean(np.all(result.extracted_face_colors == blue, axis=1))
    remainder_red_frac = np.mean(np.all(result.remainder_face_colors == red, axis=1))
    # The small test appendage's connector (peg + embed) makes up a bigger
    # share of its own surface than it would on a life-sized part, so this
    # is a majority check, not a near-total one.
    assert extracted_blue_frac > 0.5
    assert remainder_red_frac > 0.5
