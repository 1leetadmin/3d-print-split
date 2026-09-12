"""Tests for the paint_color bitstream decoder.

Test vectors are built with an independent encoder (mirroring
PrusaSlicer's FacetsAnnotation::get_triangle_as_string /
TriangleSelector::deserialize) rather than reusing any of the decoder's own
helpers, so a bug shared between encode and decode wouldn't hide behind a
tautological round-trip.
"""
from __future__ import annotations

from app.core.loaders.paint_color import decode_paint_color_states, dominant_paint_state


def _encode_leaf_nibbles(state: int) -> list[int]:
    if state <= 2:
        return [state << 2]
    if state <= 16:
        return [0b1100, state - 3]
    ext = state - 17
    return [0b1100, 0b1110, ext & 0xF, (ext >> 4) & 0xF]


def _encode_node_nibbles(node) -> list[int]:
    kind = node[0]
    if kind == "leaf":
        return _encode_leaf_nibbles(node[1])
    children = node[1]
    num_split_sides = len(children) - 1
    nibbles = [num_split_sides]
    for child in children:
        nibbles.extend(_encode_node_nibbles(child))
    return nibbles


def _encode(node) -> str:
    nibbles = _encode_node_nibbles(node)
    return "".join(format(n, "X") for n in reversed(nibbles))


def test_simple_leaf_states_match_observed_real_world_values():
    # Taken directly from a real Bambu Studio project 3MF: single-character
    # paint_color values "4" and "8" on a model whose filament list is
    # [white, yellow, orange, near-black] (Extruder1..4).
    assert dominant_paint_state("4") == 1
    assert dominant_paint_state("8") == 2


def test_encode_decode_leaf_low_states():
    for state in (0, 1, 2):
        encoded = _encode(("leaf", state))
        assert dominant_paint_state(encoded) == state


def test_encode_decode_leaf_mid_states():
    for state in (3, 5, 10, 16):
        encoded = _encode(("leaf", state))
        assert dominant_paint_state(encoded) == state


def test_encode_decode_leaf_extended_states():
    for state in (17, 42, 255):
        encoded = _encode(("leaf", state))
        assert dominant_paint_state(encoded) == state


def _summed_weights(hex_string: str) -> dict[int, float]:
    totals: dict[int, float] = {}
    for state, weight in decode_paint_color_states(hex_string):
        totals[state] = totals.get(state, 0.0) + weight
    return totals


def test_split_two_children_area_weighted():
    node = ("split", [("leaf", 1), ("leaf", 2)])
    encoded = _encode(node)
    weighted = _summed_weights(encoded)
    assert weighted[1] == 0.5
    assert weighted[2] == 0.5


def test_split_three_children_uneven_majority():
    # Majority-area state (2 of 3 equal shares) should be dominant.
    node = ("split", [("leaf", 2), ("leaf", 5), ("leaf", 5)])
    encoded = _encode(node)
    assert dominant_paint_state(encoded) == 5


def test_nested_split():
    node = (
        "split",
        [
            ("leaf", 1),
            ("split", [("leaf", 2), ("leaf", 2), ("leaf", 3), ("leaf", 3)]),
        ],
    )
    encoded = _encode(node)
    weighted = _summed_weights(encoded)
    # Root splits into 2 halves; the second half splits again into 4
    # quarters of itself, so states 2 and 3 each own 2 * (1/2 * 1/4) = 1/4.
    assert weighted[1] == 0.5
    assert weighted[2] == 0.25
    assert weighted[3] == 0.25


def test_empty_string_is_unpainted():
    assert dominant_paint_state("") == 0
