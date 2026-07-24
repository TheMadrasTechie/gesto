"""
Training.

Two model types, matched to what the gesture actually is:

- static   : one frame per sample. The gesture IS a held shape or posture
             (hand signs, letters, stances). A small Dense network.
- sequence : a window of frames. The gesture is a MOTION (waving, clapping,
             jogging). A stacked LSTM.

Both auto-select a lighter architecture on small datasets — an oversized model
on a few dozen samples overfits and collapses to predicting one class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import artifacts, data
from .regions import REGION_INFO, feature_dim

# datasets smaller than this get the lighter model
SMALL_STATIC = 300
SMALL_SEQUENCE = 100


def _keras():
    """A working Keras module, robust to how the environment exposes it."""
    from ._compat import keras
    return keras()


def _class_weights(y, n_classes) -> dict[int, float]:
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(float)
    total = counts.sum()
    return {i: float(total / (n_classes * c)) if c else 1.0
            for i, c in enumerate(counts)}


def build_static(dim: int, n_classes: int, small: bool):
    keras = _keras()
    L = keras.layers
    if small:
        layers = [L.Dense(64, activation="relu"), L.Dropout(0.3),
                  L.Dense(32, activation="relu"), L.Dropout(0.2)]
    else:
        layers = [L.Dense(256, activation="relu"), L.Dropout(0.4),
                  L.Dense(128, activation="relu"), L.Dropout(0.3),
                  L.Dense(64, activation="relu"), L.Dropout(0.2)]
    model = keras.Sequential([keras.Input((dim,)), *layers,
                              L.Dense(n_classes, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_sequence(seq_len: int, dim: int, n_classes: int, small: bool):
    keras = _keras()
    L = keras.layers
    if small:
        layers = [L.LSTM(32, activation="tanh"), L.Dropout(0.4),
                  L.Dense(32, activation="relu"), L.Dropout(0.3)]
    else:
        layers = [L.LSTM(64, return_sequences=True, activation="tanh"),
                  L.LSTM(128, return_sequences=True, activation="tanh"),
                  L.LSTM(64, activation="tanh"),
                  L.Dense(64, activation="relu"),
                  L.Dense(32, activation="relu")]
    model = keras.Sequential([keras.Input((seq_len, dim)), *layers,
                              L.Dense(n_classes, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _fit(model, X, y, labels, *, epochs, batch_size, val_split, quiet):
    keras = _keras()
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True)
    model.fit(X, y, validation_split=val_split, epochs=epochs,
              batch_size=batch_size, callbacks=[stop],
              class_weight=_class_weights(y, len(labels)),
              verbose=0 if quiet else 2)
    return model


def _report(counts: dict[str, int], quiet: bool) -> None:
    if quiet:
        return
    for name, c in counts.items():
        print(f"   {name:16} {c}")
    values = list(counts.values())
    if values and min(values) and max(values) >= 3 * min(values):
        print("Note: classes are imbalanced — applying class weights.")


def train(project_dir: str | Path, region: str, mode: str, *,
          root: str | Path = artifacts.DEFAULT_ROOT,
          seq_len: int = 30, epochs: int | None = None, batch_size: int = 16,
          val_split: float = 0.2, small: bool | None = None,
          normalized: bool = True, quiet: bool = False) -> Path:
    """Train a model and save it to a fresh, versioned artifact folder.

    Returns the run folder, e.g. artifacts/static/pose_2.
    """
    if mode not in artifacts.MODES:
        raise ValueError(f"mode must be one of {artifacts.MODES}")
    dim = feature_dim(region)

    if mode == "static":
        X, y, labels = data.load_static(project_dir, region)
        auto_small = len(X) < SMALL_STATIC
        epochs = 200 if epochs is None else epochs
    else:
        X, y, labels = data.load_sequence(project_dir, region, seq_len)
        auto_small = len(X) < SMALL_SEQUENCE
        epochs = 300 if epochs is None else epochs

    use_small = auto_small if small is None else small
    counts = data.class_summary(y, labels)

    if not quiet:
        shape = f"{len(X)} samples" if mode == "static" else f"{len(X)} sequences"
        print(f"{mode} / {region} (dim {dim}): {shape}, classes={labels}")
    _report(counts, quiet)
    if min(counts.values()) == 0:
        raise ValueError("A class has no usable samples — capture some for "
                         "every class, or lower seq_len.")
    if use_small and small is None and not quiet:
        print(f"Note: small dataset — using the lighter model (a large one "
              f"overfits and collapses on this much data).")

    rng = np.random.default_rng(42)
    order = rng.permutation(len(X))
    X, y = X[order], np.asarray(y)[order]

    if mode == "static":
        model = build_static(dim, len(labels), use_small)
    else:
        model = build_sequence(seq_len, dim, len(labels), use_small)
    if not quiet:
        model.summary()

    _fit(model, X, y, labels, epochs=epochs, batch_size=batch_size,
         val_split=val_split, quiet=quiet)

    run = artifacts.new_run(root, mode, region)
    model.save(artifacts.model_path(run))
    gesto_region, hands = REGION_INFO[region][1], REGION_INFO[region][2]
    meta = {
        "labels": labels,
        "region": region,
        "mode": mode,
        "input_dim": dim,
        "gesto_region": gesto_region,
        "hands": hands,
        "normalized": normalized,
        "samples": int(len(X)),
        "counts": counts,
    }
    if mode == "sequence":
        meta["seq_len"] = int(seq_len)
    artifacts.save_meta(run, meta)

    if not quiet:
        loss, acc = model.evaluate(X, y, verbose=0)
        print(f"\nTraining-set accuracy: {acc:.3f}")
        print(f"Saved -> {run}")
    return run
