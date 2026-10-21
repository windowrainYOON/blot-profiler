"""Project files: everything about an analysis in one file.

* ``.blotproj`` — a zip with ``project.json`` and, by default, the blot and
  marker images under ``images/`` so the project opens on another computer.
* ``.json`` — the same ``project.json`` without images (older versions wrote
  only this; they still open).

Image references keep the absolute path, the path relative to the project
file and, when embedded, the name inside the archive; :func:`resolve` tries
them in that order of reliability.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from typing import Dict, Optional, Tuple

PROJECT_EXT = ".blotproj"
EMBED_WARN_BYTES = 500 * 1024 * 1024  # ask before embedding more than this


def is_project_file(p: str) -> bool:
    p = p.lower()
    return p.endswith(PROJECT_EXT) or p.endswith(".json")


def file_ref(path: Optional[str], project_path: str, arc_name: Optional[str] = None) -> Optional[dict]:
    """How a project remembers an image file."""
    if not path:
        return None
    ref = {"path": os.path.abspath(path), "name": os.path.basename(path)}
    try:
        ref["rel"] = os.path.relpath(os.path.abspath(path), os.path.dirname(os.path.abspath(project_path)))
    except ValueError:  # another drive on Windows
        pass
    if arc_name:
        ref["file"] = arc_name
    return ref


def resolve(ref: Optional[dict], project_path: str, embedded_dir: Optional[str]) -> Optional[str]:
    """Find a remembered image: embedded copy, original path, relative path, same folder."""
    if not ref:
        return None
    if isinstance(ref, str):
        ref = {"path": ref, "name": os.path.basename(ref)}
    folder = os.path.dirname(os.path.abspath(project_path))
    cands = []
    if embedded_dir and ref.get("file"):
        cands.append(os.path.join(embedded_dir, ref["file"]))
    if ref.get("path"):
        cands.append(ref["path"])
    if ref.get("rel"):
        cands.append(os.path.normpath(os.path.join(folder, ref["rel"])))
    if ref.get("name"):
        cands.append(os.path.join(folder, ref["name"]))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def archive_names(paths) -> Dict[str, str]:
    """Unique names under images/ for the files to embed: {source path: archive name}."""
    out, used = {}, set()
    for p in paths:
        if not p or p in out:
            continue
        base = os.path.basename(p)
        stem, ext = os.path.splitext(base)
        name, k = base, 2
        while name.lower() in used:
            name = f"{stem}_{k}{ext}"
            k += 1
        used.add(name.lower())
        out[p] = f"images/{name}"
    return out


def write(path: str, data: dict, files: Dict[str, str]):
    """Write a project. ``files`` = {source path: archive name}; ignored for .json."""
    text = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = path + ".saving"
    if path.lower().endswith(PROJECT_EXT):
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json", text)
            for src, arc in files.items():
                z.write(src, arc)
    else:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
    os.replace(tmp, path)  # never leave a half-written project behind


def read(path: str) -> Tuple[dict, Optional[str]]:
    """Read a project; returns (data, folder with the extracted images or None)."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("project.json").decode("utf-8"))
            names = [n for n in z.namelist() if n.startswith("images/") and not n.endswith("/")]
            if not names:
                return data, None
            out = tempfile.mkdtemp(prefix="blotproj_")
            for n in names:
                target = os.path.normpath(os.path.join(out, n))
                if not target.startswith(os.path.normpath(out) + os.sep):  # no paths outside the folder
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(n) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            return data, out
    with open(path, encoding="utf-8") as f:
        return json.load(f), None
