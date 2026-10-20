# PyInstaller spec for Windows. Build with:  build_windows.bat
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, ROOT)

hidden = ["blot_profiler.figure", "blot_profiler.figure_window"]
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
    name="BlotProfiler",
    console=False,
    icon=os.path.join(ROOT, "packaging", "BlotProfiler.ico"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="BlotProfiler")
