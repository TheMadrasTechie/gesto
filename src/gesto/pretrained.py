"""
Download and cache pre-trained demo models.

Lets someone try detection without training anything first:

    gesto detect sequence hands_one --version default

The model is downloaded once to a local cache and reused after that.
Currently only sequence/hands_one has a hosted demo model.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

# (mode, region) -> URL of a zip containing model.keras + labels.json
PRETRAINED = {
    ("sequence", "hands_one"):
        "https://github.com/TheMadrasTechie/gesto/releases/download/"
        "gesto-labeller-v-1.0.2/hands_one.zip",
}


def _cache_root() -> Path:
    """Where downloaded demo models live (override with GESTO_CACHE)."""
    env = os.environ.get("GESTO_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "gesto" / "pretrained"


def available(mode: str, region: str) -> bool:
    return (mode, region) in PRETRAINED


def ensure_pretrained(mode: str, region: str) -> Path:
    """Return a local run folder for the demo model, downloading if needed.

    The folder contains model.keras and labels.json, exactly like a trained
    run, so the normal Predictor can load it.
    """
    key = (mode, region)
    if key not in PRETRAINED:
        have = ", ".join(f"{m}/{r}" for m, r in PRETRAINED) or "none"
        raise FileNotFoundError(
            f"No pre-trained model for {mode}/{region}. Available: {have}. "
            f"Train your own with: gesto train {mode} {region} <project_dir>")

    dest = _cache_root() / mode / region
    model = dest / "model.keras"
    labels = dest / "labels.json"

    # already downloaded and looks complete -> reuse
    if model.exists() and labels.exists():
        return dest

    url = PRETRAINED[key]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pre-trained {mode}/{region} model...\n  {url}")
    data = _download(url)

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        _extract_run(z, dest)

    if not (model.exists() and labels.exists()):
        raise FileNotFoundError(
            "Downloaded archive did not contain model.keras and labels.json. "
            f"Extracted into {dest}: {[p.name for p in dest.iterdir()]}")
    print(f"Ready: {dest}")
    return dest


def _download(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "gesto"})
    with urlopen(req) as r:                       # noqa: S310 (trusted URL)
        return r.read()


def _extract_run(z: zipfile.ZipFile, dest: Path) -> None:
    """Extract model.keras and labels.json into dest, flattening any top folder.

    The zip may hold the files at the root or inside a single folder (e.g.
    hands_one/model.keras). We copy the two files we need to dest directly.
    """
    wanted = ("model.keras", "labels.json")
    for info in z.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        if name in wanted:
            with z.open(info) as src:
                (dest / name).write_bytes(src.read())
