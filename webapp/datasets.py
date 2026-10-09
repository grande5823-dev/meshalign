"""Dataset listing + upload for the meshalign console webapp.

A *dataset id* is a stable path relative to one of the allowed roots:
``demo/…``, ``mech_parts/…`` (the repo's ``data/`` directories) and
``uploads/…`` (user uploads). Domain logic only — no FastAPI imports.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile

from alignment import ALL_EXTS, CLOUD_EXTS, MESH_EXTS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")

UPLOADS_DIR = os.environ.get(
    "MESHALIGN_UPLOADS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads"))

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def _kind_for(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext in MESH_EXTS:
        return "mesh"
    if ext in CLOUD_EXTS:
        return "cloud"
    return "unknown"


def resolve(dataset_id: str) -> str:
    """Absolute path for a dataset id; raises KeyError on traversal/unknown.

    Ids under the repo ``data/`` root are bare relative paths
    (``mech_parts/Connector.stl``, ``mech_parts/Plug.stl``); uploads carry the
    ``uploads/`` prefix.
    """
    if not re.fullmatch(r"[A-Za-z0-9_\-./ ]+", dataset_id):
        raise KeyError(dataset_id)
    if ".." in dataset_id or dataset_id.startswith("/"):
        raise KeyError(dataset_id)
    roots = [("uploads", UPLOADS_DIR), ("", DATA_ROOT)]
    for prefix, root in roots:
        if prefix:
            if not (dataset_id == prefix or dataset_id.startswith(prefix + "/")):
                continue
            rel = dataset_id[len(prefix):].lstrip("/")
        else:
            if dataset_id.startswith("/"):
                continue
            rel = dataset_id
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(os.path.realpath(root) + os.sep):
            raise KeyError(dataset_id)
        if not os.path.isfile(path):
            raise KeyError(dataset_id)
        return path
    raise KeyError(dataset_id)


def list_groups() -> list[dict]:
    """Folders usable as batch inputs: ``{id, n_files}`` (data subfolders,
    ``uploads`` and each ``uploads/<folder>``)."""
    counts: dict[str, int] = {}
    for d in list_datasets():
        counts[d["group"]] = counts.get(d["group"], 0) + 1
    return [{"id": g, "n_files": n} for g, n in sorted(counts.items())]


def resolve_group(group_id: str) -> str:
    """Absolute folder for a dataset group id (``mech_parts``, ``uploads``,
    ``uploads/<folder>``); raises KeyError on traversal/unknown."""
    if not re.fullmatch(r"[A-Za-z0-9_\-./ ]+", group_id) or ".." in group_id             or group_id.startswith("/"):
        raise KeyError(group_id)
    if group_id == "uploads" or group_id.startswith("uploads/"):
        root, rel = UPLOADS_DIR, group_id[len("uploads"):].lstrip("/")
    elif group_id == "data":
        root, rel = DATA_ROOT, ""
    else:
        root, rel = DATA_ROOT, group_id
    path = os.path.realpath(os.path.join(root, rel))
    rroot = os.path.realpath(root)
    if not (path == rroot or path.startswith(rroot + os.sep)) or not os.path.isdir(path):
        raise KeyError(group_id)
    return path


def list_datasets() -> list[dict]:
    """Every loadable geometry file under data/ and uploads/, newest first."""
    out = []

    def walk(root: str, prefix: str):
        if not os.path.isdir(root):
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            rel = os.path.relpath(dirpath, root)
            if rel == ".":
                pre = prefix
            else:
                pre = (f"{prefix}/{rel}" if prefix else rel).replace(os.sep, "/")
            for fn in sorted(filenames):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in ALL_EXTS:
                    continue
                full = os.path.join(dirpath, fn)
                out.append({
                    "id": f"{pre}/{fn}" if pre else fn,
                    "name": fn,
                    "kind": _kind_for(fn),
                    "size_bytes": os.path.getsize(full),
                    "group": prefix,
                })

    walk(DATA_ROOT, "")
    walk(UPLOADS_DIR, "uploads")
    for item in out:
        # group = the folder the file lives in (batch pairs files by folder)
        item["group"] = item["id"].rsplit("/", 1)[0] if "/" in item["id"] else "data"
    out.sort(key=lambda d: (d["group"].lower(), d["name"].lower()))
    return out


def save_upload(filename: str, fileobj, folder: str | None = None) -> dict:
    """Persist an uploaded geometry file into the uploads dir. Returns the
    dataset record. Raises ValueError on bad names/oversize/bad extension."""
    filename = os.path.basename(filename or "")
    if not filename:
        raise ValueError("empty filename")
    if not re.fullmatch(r"[A-Za-z0-9_\-./ ]+", filename):
        raise ValueError("filename may contain letters, digits, space, "
                         "- _ . / only")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALL_EXTS:
        raise ValueError(f"unsupported extension {ext!r}; expected one of "
                         f"{', '.join(ALL_EXTS)}")
    target_dir = UPLOADS_DIR
    rel_prefix = "uploads/"
    if folder:
        folder = folder.strip().strip("/")
        if not re.fullmatch(r"[A-Za-z0-9_\- ]+", folder):
            raise ValueError("folder may contain letters, digits, space, - _ only")
        target_dir = os.path.join(UPLOADS_DIR, folder)
        rel_prefix = f"uploads/{folder}/"
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, filename)
    if os.path.exists(dest):
        stem, e = os.path.splitext(filename)
        i = 1
        while os.path.exists(dest):
            dest = os.path.join(target_dir, f"{stem}-{i}{e}")
            i += 1
    size = 0
    tmp = tempfile.NamedTemporaryFile(dir=target_dir, delete=False,
                                      suffix=".part")
    try:
        with tmp:
            while True:
                chunk = fileobj.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError(f"file larger than {MAX_UPLOAD_BYTES // (1024*1024)} MB")
                tmp.write(chunk)
        # rename only after the handle is closed: Windows refuses to move an
        # open file (WinError 32)
        os.replace(tmp.name, dest)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return {"id": rel_prefix + os.path.basename(dest),
            "name": os.path.basename(dest),
            "kind": _kind_for(dest),
            "size_bytes": size,
            "group": rel_prefix.rstrip("/")}
