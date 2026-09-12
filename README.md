# STL Color Splitter

A desktop app that takes a full-color 3D model and splits it into one
watertight STL per color, so you can print each color separately on a
single-extruder printer (no AMS / multi-material unit needed) and glue the
parts back together afterward.

## How it works

1. **Load** a colored model: `.3mf` (multi-object or basematerials/pid
   per-triangle colors), `.obj`+`.mtl`, `.ply` (per-vertex color), or a
   binary `.stl` using the non-standard RGB555 color-attribute extension
   (older Cura, Simplify3D, Materialise Magics).
2. **Quantize** colors down to a manageable palette (K-means if the source
   has near-continuous painted colors; exact passthrough if it already has a
   handful of distinct colors).
3. **Voxelize** the model and label every solid voxel with the color of the
   nearest point on the original surface -- the same approach slicers use to
   decide which extruder should fill interior volume under a painted region.
4. **Reconstruct** one watertight solid per color via marching cubes, so
   every part is guaranteed printable (closed, manifold, correctly-oriented
   normals) even if the source paint boundaries weren't themselves closed
   surfaces.
5. **Export** each color as its own STL, plus a `manifest.json` and a
   combined `assembly_preview.ply` you can open in MeshLab/Blender to check
   that the parts still line up before you print.

Objects that are already separate in the source file (e.g. one `<object>`
per filament in a multi-color 3MF) are voxelized independently rather than
merged first -- two parts placed touching or coincident with each other
(the normal case for an already-split model) would otherwise create a
non-manifold seam that confuses solid voxelization.

Because parts are reconstructed from the original mesh's own coordinates,
printing each one in its original orientation and gluing matching surfaces
back together reproduces the assembled model -- no extra alignment pins are
generated (see Limitations).

## Download (Windows, no Python needed)

Go to the [Releases page](https://github.com/1leetadmin/3d-print-split/releases)
and download `STLColorSplitter.exe` from the latest release -- it's a single
portable executable, nothing to install or extract, just run it. The download
is large (~1.5 GB, mostly VTK/Qt, all bundled into that one file) and it takes
a few seconds to start each time since it self-extracts to a temp folder on
launch -- no Python install required either way.

Every push of a `v*` tag builds a fresh Windows release automatically (see
`.github/workflows/build-windows.yml`); you can also trigger a build manually
from the repo's Actions tab without cutting a release.

## Install from source

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### GUI

```bash
python -m app.main
```

Open a model, adjust voxel resolution / max color count if needed, click
**Split into parts**, then **Export all parts…** to write STL files.

### CLI (no display needed)

```bash
python -m scripts.cli model.3mf --out-dir split_output --voxels 120 --max-colors 8
```

Run `python -m scripts.cli --help` for all options.

## Choosing voxel resolution

Higher "voxels along longest axis" gives more accurate part boundaries and
volumes at the cost of slower processing and more memory. Voxelization
inherently pads each part's surface by roughly one voxel, so volumes/edges
converge to the true geometry only as resolution increases -- 120-200 is a
reasonable starting point for most desktop models; drop to 60-80 for a quick
preview, or go higher for parts with fine color detail.

## Limitations

- **No sub-triangle paint decoding.** PrusaSlicer/Bambu Studio's proprietary
  "paint on supports/seams/multi-material" per-triangle segmentation format
  is not decoded. If your 3MF only carries color via that painting feature,
  re-export as separate colored objects/materials, or as a vertex-colored
  OBJ/PLY, before loading it here.
- **No texture sampling for photographic textures** beyond what trimesh's
  best-effort UV-to-vertex-color conversion provides; solid painted/material
  colors are the primary supported case.
- **No auto-generated alignment features.** Parts are exported in their
  original positions/orientations with no registration pins, dowel holes, or
  keys added at the seams -- you're gluing flat-to-flat against the original
  surface.
- **Legacy color-STL bit order is a convention, not a standard.** If colors
  look like red and blue are swapped after loading a color-STL file, use the
  "Swap R/B" option (GUI checkbox or `--swap-rb` CLI flag).

## Project layout

```
app/
  core/            # pure-Python pipeline: no Qt/GUI dependencies
    model.py         # ColoredMesh / ColorPart data classes
    loaders/         # per-format loaders + format auto-detect
    colors.py        # color quantization
    voxelize.py      # voxelization + nearest-surface color labeling
    split.py         # per-color marching-cubes reconstruction
    export.py        # STL + manifest + preview export
  gui/             # PySide6 + PyVista desktop UI
  main.py          # GUI entry point
scripts/
  cli.py           # headless CLI entry point
tests/             # pytest suite for app/core (no GUI/display required)
packaging/
  app.spec         # PyInstaller spec used by the Windows build workflow
.github/workflows/
  build-windows.yml  # builds + releases the Windows .exe
```

## Building the Windows executable yourself

```bash
pip install -r requirements.txt pyinstaller
pyinstaller packaging/app.spec --noconfirm
```

The portable executable is written to `dist/STLColorSplitter.exe` -- copy that
one file anywhere to distribute it.

## Running tests

```bash
pip install pytest
pytest tests/
```
