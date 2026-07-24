"""
Loading Gesto Labeller datasets.

A Gesto project folder looks like:

    <project>/
        project.json
        data/
            static/<label>/<uid>.npy      each (D,)
            sequence/<label>/<uid>.npy    each (T, D)

Use the "Copy path" button on a project card in Gesto Labeller to get the path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .regions import feature_dim


def project_meta(project_dir: str | Path) -> dict:
    """Contents of project.json, or {} when it isn't there."""
    p = Path(project_dir) / "project.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _class_dirs(project_dir: str | Path, mode: str) -> tuple[Path, list[str]]:
    root = Path(project_dir) / "data" / mode
    if not root.exists():
        other = "sequence" if mode == "static" else "static"
        hint = ""
        if (Path(project_dir) / "data" / other).exists():
            hint = (f"\nThis project has {other} data instead — either capture "
                    f"in {mode.capitalize()} mode in Gesto Labeller, or train "
                    f"the {other} model.")
        raise FileNotFoundError(f"No {mode} data at {root}{hint}")
    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not labels:
        raise ValueError(f"No class folders under {root}")
    return root, labels


def load_static(project_dir: str | Path, region: str):
    """-> X (N, D) float32, y (N,) int, labels list[str]."""
    dim = feature_dim(region)
    root, labels = _class_dirs(project_dir, "static")
    index = {name: i for i, name in enumerate(labels)}

    X: list[np.ndarray] = []
    y: list[int] = []
    for name in labels:
        for f in sorted((root / name).glob("*.npy")):
            arr = np.load(f).astype(np.float32).reshape(-1)
            if arr.shape[0] != dim:
                raise ValueError(
                    f"{f} has dim {arr.shape[0]} but region {region!r} expects "
                    f"{dim}. Wrong region for this project?")
            X.append(arr)
            y.append(index[name])
    if not X:
        raise ValueError(f"No .npy samples found under {root}")
    return np.asarray(X, np.float32), np.asarray(y), labels


def load_sequence(project_dir: str | Path, region: str, seq_len: int = 30):
    """-> X (N, seq_len, D) float32, y (N,) int, labels list[str].

    Sequences shorter than seq_len are skipped (they can't fill the window);
    longer ones are trimmed to the first seq_len frames. Capture at a consistent
    length in Gesto Labeller ("Max frames") for best results.
    """
    dim = feature_dim(region)
    root, labels = _class_dirs(project_dir, "sequence")
    index = {name: i for i, name in enumerate(labels)}

    X: list[np.ndarray] = []
    y: list[int] = []
    skipped = 0
    for name in labels:
        for f in sorted((root / name).glob("*.npy")):
            arr = np.load(f).astype(np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[1] != dim:
                raise ValueError(
                    f"{f} has dim {arr.shape[1]} but region {region!r} expects "
                    f"{dim}. Wrong region for this project?")
            if arr.shape[0] < seq_len:
                skipped += 1
                continue
            X.append(arr[:seq_len])
            y.append(index[name])
    if not X:
        raise ValueError(
            f"No sequences with at least {seq_len} frames under {root}. "
            f"Lower seq_len or capture longer clips.")
    if skipped:
        print(f"Note: skipped {skipped} sequence(s) shorter than {seq_len} frames.")
    return np.asarray(X, np.float32), np.asarray(y), labels


def class_summary(y, labels) -> dict[str, int]:
    counts = np.bincount(np.asarray(y), minlength=len(labels))
    return {name: int(c) for name, c in zip(labels, counts)}
