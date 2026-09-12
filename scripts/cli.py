#!/usr/bin/env python3
"""Headless CLI for the color-splitting pipeline (no GUI required).

Usage:
    python -m scripts.cli input.3mf --out-dir out/ --voxels 120 --max-colors 8
"""
from __future__ import annotations

import argparse
import sys
import time

from app.core.export import export_parts
from app.core.loaders.detect import load_colored_mesh
from app.core.split import split_by_color


def _progress(msg: str, frac: float) -> None:
    print(f"[{frac * 100:5.1f}%] {msg}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to a .3mf, .obj, .ply, or color .stl file")
    parser.add_argument("--out-dir", default="split_output", help="Output directory")
    parser.add_argument(
        "--voxels", type=int, default=120, help="Voxels along the model's longest axis"
    )
    parser.add_argument(
        "--max-colors", type=int, default=8, help="Maximum distinct color parts to produce"
    )
    parser.add_argument(
        "--swap-rb",
        action="store_true",
        help="Swap red/blue when decoding legacy color-STL attribute bytes",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    print(f"Loading {args.input} ...", file=sys.stderr)
    colored_mesh = load_colored_mesh(args.input, swap_rb=args.swap_rb)
    print(
        f"Loaded {len(colored_mesh.faces)} faces, "
        f"{len(colored_mesh.unique_colors)} distinct source colors",
        file=sys.stderr,
    )

    parts = split_by_color(
        colored_mesh,
        voxels_along_longest=args.voxels,
        max_colors=args.max_colors,
        progress_cb=_progress,
    )
    if not parts:
        print("No parts produced -- check the input file has color data.", file=sys.stderr)
        return 1

    manifest_path = export_parts(parts, args.out_dir)
    dt = time.time() - t0
    print(f"\nWrote {len(parts)} part(s) to {args.out_dir} in {dt:.1f}s")
    for part in parts:
        watertight_note = "" if part.voxel_count else " (empty, skipped)"
        print(f"  #{part.hex_color}  {part.voxel_count} voxels{watertight_note}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
