# PyInstaller spec for the STL Color Splitter desktop app.
#
# Build from the repo root with:
#   pyinstaller packaging/app.spec
#
# Builds a single portable executable (onefile): everything needed to run is
# packed into one .exe that self-extracts to a temp directory on each launch
# (a few seconds slower to start than an onedir build, but the download is
# one file with nothing to unzip).
#
# VTK/PyVista and PySide6 both dynamically pull in a lot of submodules and
# plugin files that PyInstaller's static import scan can't see, so we pull
# them in wholesale with collect_all rather than chasing hidden-import
# errors one at a time.
import os

from PyInstaller.utils.hooks import collect_all

SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
REPO_ROOT = os.path.dirname(SPEC_DIR)

datas = []
binaries = []
hiddenimports = []

for pkg in ("vtkmodules", "pyvista", "pyvistaqt", "PySide6", "trimesh", "skimage"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

block_cipher = None

a = Analysis(
    [os.path.join(REPO_ROOT, "app", "main.py")],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    exclude_binaries=False,
    name="STLColorSplitter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
