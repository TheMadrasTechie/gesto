"""
Small compatibility shims.

Keras has moved around across TensorFlow versions: standalone `keras` (Keras 3,
used by TF 2.16+), `tensorflow.keras`, and `from tensorflow import keras`. A
broken or partially-upgraded TensorFlow install makes one or more of these fail
with "cannot import name 'keras' from 'tensorflow' (unknown location)".

`keras()` tries them in turn and, if all fail, raises one clear message instead
of a confusing traceback.
"""

from __future__ import annotations

import importlib


def keras():
    """Return a working Keras module, however this environment exposes it."""
    errors = []

    # 1. standalone Keras 3 (what TF 2.16+ actually uses under the hood)
    try:
        return importlib.import_module("keras")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import keras -> {exc}")

    # 2. tensorflow.keras submodule
    try:
        return importlib.import_module("tensorflow.keras")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import tensorflow.keras -> {exc}")

    # 3. attribute on the tensorflow module
    try:
        import tensorflow as tf
        if hasattr(tf, "keras"):
            return tf.keras
        errors.append("tensorflow has no attribute 'keras'")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import tensorflow -> {exc}")

    detail = "\n  ".join(errors)
    raise ImportError(
        "Could not import Keras. Your TensorFlow install looks broken or "
        "incomplete — this often happens after up/downgrading TensorFlow in "
        "place. Reinstall it cleanly:\n"
        '    pip install --force-reinstall "tensorflow==2.17.1"\n'
        "or, best, start a fresh virtual environment and `pip install gesto`.\n"
        f"Tried:\n  {detail}"
    )
