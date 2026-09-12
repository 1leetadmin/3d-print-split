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

The window is split into two views: the **left pane always shows the
untouched original model**, unaffected by anything you do, so you always
have something to compare against; the **right pane is your working
copy**, where clicking, selecting and extracting all happen and update in
real time.

1. **Load** a colored model: `.3mf` (Bambu Studio/OrcaSlicer/PrusaSlicer
   paint-tool color data, multi-object per-filament files, or
   basematerials/pid per-triangle materials), `.obj`+`.mtl`, `.ply`
   (per-vertex color), or a binary `.stl` using the non-standard RGB555
   color-attribute extension (older Cura, Simplify3D, Materialise Magics).
2. The model is prepared **once** into a single starting solid, kept at its
   **original resolution** whenever possible -- an already-watertight mesh
   is used as-is, one with only tiny local topology defects (e.g.
   sub-millimeter artifacts some slicers leave behind at painted color
   boundaries) gets those welded shut in place, and one with genuine small
   gaps gets a surface-preserving hole-fill. None of these regenerate the
   surface, so the result looks exactly as smooth as the source model. Only
   a mesh too damaged for all of that falls back to a voxel remesh, which is
   watertight-by-construction but blockier at the scale of the chosen
   fallback resolution.
3. **Click** any point on the model in the right pane. That grows outward
   from the clicked face into neighboring faces whose color stays within
   your chosen tolerance, instantly highlighting the patch that would be
   cut -- changing the tolerance re-highlights immediately, before you
   commit to anything.
4. **Extract selected part**: the highlighted patch is cut away from the
   rest, keeping essentially all of each resulting piece's surface as the
   original geometry -- new triangles only appear right at the seam and the
   connector, so the cut stays smooth. A correctly-sized peg-and-socket
   connector is fused into the cut directly, analytically, at the seam
   (sized and oriented from the seam's own footprint), and both resulting
   solids are verified watertight before being accepted. If the selection's
   boundary is too irregular for that (e.g. a highly pixelated or
   self-touching patch), extraction automatically falls back to a
   voxel-based cut instead -- blockier at the seam, but always correct and
   watertight; this never silently produces broken geometry either way. The
   extracted part is placed to the side of the shrinking body in the right
   pane so you can see the cut worked correctly.
5. Repeat on the remaining body for the next part, **undo/redo** any cut,
   and **Export all parts** once you're done.

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

**Tag names are the release filename**, so tag with what changed, not just a
bare version -- e.g. `v1.1.0-click-to-extract-connectors`, not `v1.1.0`. A
manual (non-tag) build instead asks for a short description and bakes that,
plus the date and commit, into the filename -- there's no generic
"latest build" artifact name anywhere in this pipeline.

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

Open a model, adjust max colors / click tolerance / connector style as
needed (the "fallback repair/remesh resolution" only matters for a source
mesh too damaged to keep at full resolution, or a selection boundary too
irregular for the surface-preserving cut), click a part in the right pane
to select it, **Extract selected part**, repeat, then **Export all
parts…**.

### CLI (bulk split, no connectors, no display needed)

For quickly splitting every color into its own file with no interactive
step and no connectors:

```bash
python -m scripts.cli model.3mf --out-dir split_output --voxels 120 --max-colors 8
```

Run `python -m scripts.cli --help` for all options.

## Choosing the fallback resolution

The "fallback repair/remesh resolution" setting only comes into play when a
cut *can't* be made by preserving the original surface -- an unusually
damaged source mesh, or a selection boundary too irregular for the
analytic cap (a highly pixelated or self-touching patch). In that case,
higher "voxels along longest axis" gives more accurate boundaries and
volumes at the cost of slower processing and more memory; voxelization
inherently pads a part's surface by roughly one voxel, so results converge
to the true geometry only as resolution increases. 120-200 is a reasonable
starting point when it does kick in; drop to 60-80 for a quick preview, or
go higher for fine color detail. It has no effect at all on a cut that
succeeds via the normal surface-preserving path, which always matches the
source model's own resolution.

## Limitations

- **A very irregular selection boundary falls back to a voxel-based cut**
  for that one extraction (blockier at the seam, still watertight and
  correct) -- e.g. a highly pixelated or self-touching color patch on a
  finely-tessellated real-world mesh. This only affects that specific cut;
  the master body and every other part stay at full resolution.
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
    meshing.py       # voxel mask -> watertight mesh; local topology-defect welding
    selection.py     # flood-fill face selection by color tolerance
    surface_cut.py   # surface-preserving cut: boundary caps + analytic connector
    extraction.py    # click-to-extract: surface cut, falling back to voxel cut
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
