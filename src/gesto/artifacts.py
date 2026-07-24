"""
Artifact storage.

Everything a training run produces lives under one root folder, split by mode
and then region:

    artifacts/
        static/
            pose/            <- first static pose model
                model.keras
                labels.json
            pose_2/          <- next one doesn't overwrite; it's versioned
            hands_one/
        sequence/
            pose/
            legs/

`new_run()` never overwrites: if `artifacts/static/pose` exists it returns
`pose_2`, then `pose_3`, and so on. `resolve()` finds the newest version when
you don't name one explicitly, so `gesto detect static pose` picks up the model
you just trained.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MODES = ("static", "sequence")
DEFAULT_ROOT = "artifacts"

_VERSION_RE = re.compile(r"^(?P<base>.+?)(?:_(?P<n>\d+))?$")


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def mode_dir(root: str | Path, mode: str) -> Path:
    _check_mode(mode)
    return Path(root) / mode


def _version_of(name: str, base: str) -> int:
    """Return the version number a folder name represents for `base`.

    "pose" -> 1, "pose_2" -> 2, anything else -> 0 (not a version of base).
    """
    m = _VERSION_RE.match(name)
    if not m or m.group("base") != base:
        return 0
    n = m.group("n")
    return int(n) if n else 1


def list_runs(root: str | Path, mode: str, region: str) -> list[Path]:
    """Every run folder for this mode+region, oldest version first."""
    d = mode_dir(root, mode)
    if not d.exists():
        return []
    runs = [(p, _version_of(p.name, region)) for p in d.iterdir() if p.is_dir()]
    runs = [(p, v) for p, v in runs if v > 0]
    runs.sort(key=lambda t: t[1])
    return [p for p, _ in runs]


def new_run(root: str | Path, mode: str, region: str) -> Path:
    """Create and return a fresh run folder, versioning instead of overwriting.

    First call -> artifacts/<mode>/<region>
    Later calls -> artifacts/<mode>/<region>_2, _3, ...
    """
    d = mode_dir(root, mode)
    d.mkdir(parents=True, exist_ok=True)
    existing = list_runs(root, mode, region)
    if not existing:
        run = d / region
    else:
        highest = max(_version_of(p.name, region) for p in existing)
        run = d / f"{region}_{highest + 1}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def resolve(root: str | Path, mode: str, region: str,
            version: int | str | None = None) -> Path:
    """Locate an existing run folder.

    version=None  -> newest version
    version=2     -> artifacts/<mode>/<region>_2
    version="pose_2" (a folder name) -> that exact folder
    """
    d = mode_dir(root, mode)
    if isinstance(version, str) and not version.isdigit():
        run = d / version
        if not run.exists():
            raise FileNotFoundError(f"No such run: {run}")
        return run

    runs = list_runs(root, mode, region)
    if not runs:
        raise FileNotFoundError(
            f"No {mode} model for region {region!r} under {d}. "
            f"Train one first: gesto train {mode} {region} <project_dir>")
    if version is None:
        return runs[-1]                      # newest
    want = int(version)
    for p in runs:
        if _version_of(p.name, region) == want:
            return p
    have = ", ".join(p.name for p in runs)
    raise FileNotFoundError(
        f"No version {want} for {mode}/{region}. Available: {have}")


def save_meta(run: Path, meta: dict) -> Path:
    path = Path(run) / "labels.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_meta(run: Path) -> dict:
    path = Path(run) / "labels.json"
    if not path.exists():
        raise FileNotFoundError(f"No labels.json in {run}")
    return json.loads(path.read_text(encoding="utf-8"))


def model_path(run: Path) -> Path:
    return Path(run) / "model.keras"
