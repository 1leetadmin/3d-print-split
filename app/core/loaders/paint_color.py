"""Decoder for the per-triangle ``paint_color`` attribute written by Bambu
Studio / PrusaSlicer / OrcaSlicer when a model is colored with the paint
tool (as opposed to being modeled as separate colored objects/materials).

This is a hex-string encoding of a bitstream describing, for each original
triangle, a quad-tree of recursive splits down to leaves each carrying a
"triangle state" -- state 0 means "unpainted, use the object's default
extruder", state N (N>=1) means "use extruder N".

The format is reverse engineered from PrusaSlicer's own (open source)
implementation -- specifically
``Slic3r::Domain::FacetsAnnotation::set_triangle_from_string`` (the hex
string <-> bitstream conversion) and
``Slic3r::Biz::Algorithms::TriangleSelector::deserialize`` (the recursive
split-tree walk) in github.com/prusa3d/PrusaSlicer. Decoding here is
intentionally simplified relative to the real algorithm: the real format
also reconstructs exact sub-triangle geometry (needed for rendering the
paint brush strokes precisely), which requires knowing which of a
triangle's 3 edges a partial split happens across. We don't need that
precision -- the voxel remeshing downstream already discretizes color at a
coarser scale than a single triangle -- so a split's children are treated
as equal-area shares of the parent instead. This can only blur a
color boundary that falls inside a single original triangle; it cannot
change which colors are present or their approximate proportions.
"""
from __future__ import annotations

NONE_STATE = 0


def _hex_nibbles_in_processing_order(hex_string: str) -> list[int]:
    """PrusaSlicer builds the bitstream by walking the hex string back to
    front, so that is also the order codes are consumed when decoding.
    """
    return [int(ch, 16) for ch in reversed(hex_string)]


def _decode_leaf_state(nibble: int, nibbles: list[int], pos: int) -> tuple[int, int]:
    if (nibble & 0b1100) != 0b1100:
        return nibble >> 2, pos
    second = nibbles[pos]
    pos += 1
    if second != 0b1110:
        return second + 3, pos
    lo = nibbles[pos]
    hi = nibbles[pos + 1]
    pos += 2
    return (lo | (hi << 4)) + 17, pos


def _decode_node(nibbles: list[int], pos: int) -> tuple[list[tuple[int, float]], int]:
    nibble = nibbles[pos]
    pos += 1
    num_split_sides = nibble & 0b11
    if num_split_sides == 0:
        state, pos = _decode_leaf_state(nibble, nibbles, pos)
        return [(state, 1.0)], pos

    num_children = num_split_sides + 1
    weighted_states: list[tuple[int, float]] = []
    for _ in range(num_children):
        child_states, pos = _decode_node(nibbles, pos)
        share = 1.0 / num_children
        weighted_states.extend((state, weight * share) for state, weight in child_states)
    return weighted_states, pos


def decode_paint_color_states(hex_string: str) -> list[tuple[int, float]]:
    """Return [(state, area_fraction), ...] for one triangle's paint_color
    string, fractions summing to 1.0. An empty string means "unpainted".
    """
    if not hex_string:
        return [(NONE_STATE, 1.0)]
    nibbles = _hex_nibbles_in_processing_order(hex_string)
    weighted_states, _pos = _decode_node(nibbles, 0)
    return weighted_states


def dominant_paint_state(hex_string: str) -> int:
    """The area-majority state for one triangle's paint_color string."""
    weighted = decode_paint_color_states(hex_string)
    if len(weighted) == 1:
        return weighted[0][0]
    totals: dict[int, float] = {}
    for state, weight in weighted:
        totals[state] = totals.get(state, 0.0) + weight
    return max(totals.items(), key=lambda kv: kv[1])[0]
