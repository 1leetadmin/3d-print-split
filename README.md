# STL Color Splitter

A desktop app for taking a full-color 3D model and cutting it into separate,
watertight, single-color pieces you can print on a single-extruder printer
(no AMS / multi-material unit needed) and glue back together afterward.

The GUI's main workflow is **interactive**: click a colored part of the
model (say, a duck's orange beak) and it cuts just that part away from the
rest, automatically generating a correctly-sized peg-and-socket connector at
the seam so the two printed pieces snap/glue back together in the right
place. Repeat for each part you want printed separately. A headless CLI is
also included for bulk "just split every color into its own file, no
connectors" batch jobs.

## How the interactive workflow works

1. **Load** a colored model: `.3mf` (Bambu Studio/OrcaSlicer/PrusaSlicer
   paint-tool color data, multi-object per-filament files, or
   basematerials/pid per-triangle materials), `.obj`+`.mtl`, `.ply`
   (per-vertex color), or a binary `.stl` using the non-standard RGB555
   color-attribute extension (older Cura, Simplify3D, Materialise Magics).
2. The whole model is voxelized and reconstructed **once** into a single
   clean watertight solid with its original colors carried over -- this
   also repairs any gaps/self-intersections in the source mesh before you
   start cutting.
3. **Click** any point on the model. That grows outward from the clicked
   face into neighboring faces whose color stays within your chosen
   tolerance, highlighting the patch that would be cut.
4. **Extract**: the volume under the highlighted patch is separated from
   the rest via the same voxelize/marching-cubes pipeline, and a
   peg-and-socket connector is baked directly into both pieces' voxel data
   at the seam -- sized from the seam's own footprint, oriented along the
   axis between the two pieces. Because it's generated at the voxel level
   (not a mesh boolean), both resulting solids are guaranteed watertight,
   the same guarantee the rest of the pipeline relies on.
5. Repeat on the remaining body for the next part, **undo/redo** any cut,
   toggle **Show original** to compare against the untouched starting
   model, and **Export all parts** once you're done.

### Extraction modes

- **Incremental** (default): each cut shrinks the body you click on next,
  so you're building up a full assembly piece by piece.
- **Independent**: every cut is made against the pristine starting model
  regardless of what's already been extracted -- simpler for testing one
  cut in isolation, but two independently-extracted parts won't know about
  each other's connectors.

### Connector styles

- **Auto** (default): round peg for small/compact seams, a D-shaped
  (keyed) peg for larger or elongated seams so the two parts can't rotate
  relative to each other once assembled.
- **Round peg + hole**: always a plain cylindrical peg.
- **Keyed (D-shaped)**: always flatten one side, forcing a single assembly
  orientation.
- **None**: just cut, no connector.

The socket is sized with a small clearance gap over the peg for an
easy hand-assembly/glue fit, not a tight press-fit.

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

### GUI (interactive click-to-extract)

```bash
python -m app.main
```

Open a model, adjust voxel resolution / max colors / click tolerance /
connector style as needed, click a part to select it, **Extract selected
part**, repeat, then **Export all parts…**.

### CLI (bulk split, no connectors, no display needed)

For quickly splitting every color into its own file with no interactive
step and no connectors:

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

- **PrusaSlicer/Bambu Studio's per-triangle paint format is decoded**, but
  only the common case where a triangle is painted as a single color
  end-to-end; the rare case of a single original triangle split into
  multiple colors by the paint tool is approximated as one dominant color
  for that triangle (voxel remeshing already discretizes at a coarser
  scale than this, so it doesn't lose meaningful detail in practice).
- **No texture sampling for photographic textures** beyond what trimesh's
  best-effort UV-to-vertex-color conversion provides for OBJ/PLY; solid
  painted/material colors are the primary supported case.
- **Connectors are only generated by the interactive GUI workflow.** The
  bulk CLI split (`scripts/cli.py`, or the underlying `split_by_color`)
  just cuts along original color boundaries with no connectors added.
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
    meshing.py       # voxel mask -> watertight mesh (shared by both pipelines)
    selection.py     # flood-fill face selection by color tolerance
    extraction.py    # click-to-extract: cut + bake peg/socket connector
    project.py       # interactive session state: body, parts, undo/redo
    split.py         # bulk per-color marching-cubes reconstruction (CLI)
    export.py        # STL + manifest export for the bulk CLI pipeline
  gui/             # PySide6 + PyVista interactive desktop UI
  main.py          # GUI entry point
scripts/
  cli.py           # headless bulk-split CLI entry point
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
