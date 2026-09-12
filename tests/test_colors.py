import numpy as np

from app.core.colors import quantize_face_colors


def test_quantize_passthrough_when_within_budget():
    face_colors = np.array(
        [[255, 0, 0]] * 5 + [[0, 0, 255]] * 3 + [[0, 255, 0]] * 2, dtype=np.uint8
    )
    palette, labels = quantize_face_colors(face_colors, max_colors=8)

    assert len(palette) == 3
    reconstructed = palette[labels]
    assert np.array_equal(reconstructed, face_colors)


def test_quantize_clusters_down_to_budget():
    rng = np.random.default_rng(0)
    # Two noisy clumps of near-red and near-blue colors -> should collapse to 2.
    reds = np.clip(np.array([200, 20, 20]) + rng.integers(-10, 10, size=(50, 3)), 0, 255)
    blues = np.clip(np.array([20, 20, 200]) + rng.integers(-10, 10, size=(50, 3)), 0, 255)
    face_colors = np.vstack([reds, blues]).astype(np.uint8)

    palette, labels = quantize_face_colors(face_colors, max_colors=2)

    assert len(palette) == 2
    assert labels.shape == (100,)
    # Each original clump should map predominantly to a single label.
    red_labels = labels[:50]
    blue_labels = labels[50:]
    assert len(np.unique(red_labels)) == 1
    assert len(np.unique(blue_labels)) == 1
    assert red_labels[0] != blue_labels[0]
