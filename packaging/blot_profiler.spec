# PyInstaller spec for the macOS app bundle. Build with:  ./build_mac.sh
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

sys.path.insert(0, os.path.abspath(os.path.join(SPECPATH, "..")))
from blot_profiler import __version__  # noqa: E402

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

hidden = []
for pkg in ("skimage.restoration", "skimage.filters", "skimage.measure", "skimage.morphology"):
    hidden += collect_submodules(pkg)
datas = collect_data_files("skimage", includes=["**/*.pyi"])

a = Analysis(
    [os.path.join(ROOT, "run_blot_profiler.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "PyQt5", "PyQt6", "IPython", "pytest", "skimage.data"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Blot Profiler",
    console=False,
    argv_emulation=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Blot Profiler")

icon = os.path.join(ROOT, "build", "BlotProfiler.icns")
app = BUNDLE(
    coll,
    name="Blot Profiler.app",
    icon=icon if os.path.exists(icon) else None,
    bundle_identifier="io.github.windowrainyoon.blotprofiler",
    version=__version__,
    info_plist={
        "CFBundleName": "Blot Profiler",
        "CFBundleDisplayName": "Blot Profiler",
        "CFBundleShortVersionString": __version__,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "TIFF image",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": ["public.tiff"],
            },
            {
                "CFBundleTypeName": "Blot Profiler analysis",
                "CFBundleTypeRole": "Editor",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": ["public.json"],
            },
        ],
    },
)
