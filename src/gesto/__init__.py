"""
gesto — train and run gesture recognition models from Gesto Labeller datasets.

Two model types:
    static    single frame; the gesture is a held shape or posture
    sequence  a window of frames; the gesture is a motion

Five regions: hands_one, hands_two, pose, legs, full.

Quick start (Python):

    import gesto

    run = gesto.train("./gesto_projects/signs", region="hands_one",
                      mode="static")
    gesto.detect("static", "hands_one")

Quick start (command line):

    gesto train static hands_one ./gesto_projects/signs
    gesto detect static hands_one

Models are saved under artifacts/<mode>/<region>/, versioned so a new run never
overwrites an old one (pose, pose_2, pose_3, ...).
"""

from __future__ import annotations

__version__ = "0.1.3"

from .regions import REGION_KEYS, REGION_INFO, feature_dim, extract, normalize

__all__ = [
    "__version__",
    "REGION_KEYS",
    "REGION_INFO",
    "feature_dim",
    "extract",
    "normalize",
    "train",
    "detect",
    "Predictor",
    "artifacts",
    "data",
]


def train(project_dir, region: str, mode: str = "static", **kwargs):
    """Train a model. See gesto.train.train for the full signature."""
    from .train import train as _train
    return _train(project_dir, region, mode, **kwargs)


def detect(mode: str, region: str, **kwargs):
    """Run live detection. See gesto.detect.run for the full signature."""
    from .detect import run as _run
    return _run(mode, region, **kwargs)


def Predictor(*args, **kwargs):  # noqa: N802 - re-exported class-like helper
    """Load a trained model for programmatic use."""
    from .detect import Predictor as _Predictor
    return _Predictor(*args, **kwargs)


def __getattr__(name: str):
    # lazy submodule access: gesto.artifacts / gesto.data without import cost
    if name in ("artifacts", "data"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
