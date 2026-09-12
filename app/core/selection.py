"""Flood-fill face selection by color, starting from a single clicked face.

This is the "magic wand" behind click-to-extract: a click gives one face,
and this grows that into a contiguous patch of faces whose color stays
within a tolerance of the clicked face's own color (not a drifting
neighbor-to-neighbor comparison, so a long, gradual color gradient won't
runaway-select the whole model).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import trimesh


def build_face_adjacency_list(mesh: trimesh.Trimesh) -> list[list[int]]:
    """One adjacency list per face, built once per mesh and reused across
    clicks -- rebuilding it per click would dominate runtime on a large mesh.
    """
    n_faces = len(mesh.faces)
    adjacency_list: list[list[int]] = [[] for _ in range(n_faces)]
    for a, b in mesh.face_adjacency:
        adjacency_list[a].append(b)
        adjacency_list[b].append(a)
    return adjacency_list


def flood_fill_by_color(
    face_colors: np.ndarray,
    adjacency_list: list[list[int]],
    seed_face: int,
    tolerance: float,
) -> np.ndarray:
    """Boolean mask (len(face_colors),) of faces reachable from seed_face by
    face adjacency without ever leaving color tolerance of the seed's own
    color. ``tolerance`` is a Euclidean RGB distance (0 = exact match only,
    ~441 = the whole color cube, i.e. everything connected gets selected).
    """
    n_faces = len(face_colors)
    selected = np.zeros(n_faces, dtype=bool)
    seed_color = face_colors[seed_face].astype(np.int32)

    def within_tolerance(idx: int) -> bool:
        diff = face_colors[idx].astype(np.int32) - seed_color
        return float(np.sqrt(np.dot(diff, diff))) <= tolerance

    selected[seed_face] = True
    queue = deque([seed_face])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency_list[current]:
            if not selected[neighbor] and within_tolerance(neighbor):
                selected[neighbor] = True
                queue.append(neighbor)

    return selected
