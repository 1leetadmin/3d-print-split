"""Color quantization: collapse near-continuous per-face colors into a small
discrete palette so the model can be split into a bounded number of
printable, single-color parts.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2


def quantize_face_colors(
    face_colors: np.ndarray, max_colors: int = 8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster per-face RGB colors down to at most ``max_colors`` groups.

    Returns ``(palette, labels)`` where ``palette`` is ``(K, 3)`` uint8 and
    ``labels`` is ``(M,)`` int giving each face's index into ``palette``.
    If the mesh already has at most ``max_colors`` distinct colors, they are
    used as-is (no clustering, no color drift).
    """
    face_colors = np.asarray(face_colors, dtype=np.uint8).reshape(-1, 3)
    unique = np.unique(face_colors, axis=0)

    if len(unique) <= max_colors:
        palette = unique
        lut = {tuple(c): i for i, c in enumerate(palette)}
        labels = np.array([lut[tuple(c)] for c in face_colors], dtype=np.int64)
        return palette, labels

    data = face_colors.astype(np.float64)
    rng = np.random.default_rng(seed)
    # Seed centroids from a random sample of observed colors for reproducibility
    # and to avoid empty clusters on flat/duplicated color data.
    seed_idx = rng.choice(len(unique), size=max_colors, replace=False)
    init = unique[seed_idx].astype(np.float64)

    centroids, labels = kmeans2(data, init, minit="matrix", seed=seed)
    palette = np.clip(np.round(centroids), 0, 255).astype(np.uint8)
    return palette, labels.astype(np.int64)


def rgb_to_hex(rgb: tuple[int, int, int] | np.ndarray) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"
