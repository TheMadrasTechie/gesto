"""
train_static.py — train a single-frame (static) gesture classifier.

Reads Gesto's static/ .npy samples (one landmark vector per sample) and trains
a small Dense network. This is the right model when each gesture is a POSE held
in one frame (e.g. a hand shape, a body stance) rather than a motion.

Outputs (into --out, default ./artifacts_static):
    static_model.keras     the trained Keras model
    labels.json            class-index -> class-name mapping + feature dim
    (optional) static_model.tflite via convert_tflite.py

Usage:
    python train_static.py /path/to/project_folder
    python train_static.py /path/to/project_folder --epochs 80 --out my_out

Matches the Keras conventions from the sign-language project (plain Sequential,
Dense stack, categorical labels).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gesto_data import load_static, load_project_meta


def build_model(input_dim, num_classes):
    """A compact MLP for single-frame landmark vectors."""
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir", help="Path to a Gesto project folder")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--out", default="artifacts_static")
    args = ap.parse_args()

    meta = load_project_meta(args.project_dir)
    X, y, labels = load_static(args.project_dir)
    if len(X) == 0:
        raise SystemExit("No static samples found. Capture some first.")

    print(f"Project '{meta.get('name')}'  region={meta.get('region')}")
    print(f"Loaded {len(X)} samples, dim={X.shape[1]}, classes={labels}")
    # warn on tiny/imbalanced data — common with hand-captured sets
    counts = np.bincount(y, minlength=len(labels))
    for name, c in zip(labels, counts):
        print(f"   {name:20} {c} samples")
    if counts.min() < 5:
        print("WARNING: some classes have <5 samples; accuracy will be unreliable.")

    # shuffle
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], np.asarray(y)[idx]

    model = build_model(X.shape[1], len(labels))
    model.summary()

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True)
    model.fit(X, y, validation_split=args.val_split,
              epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[es], verbose=2)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save(out / "static_model.keras")
    (out / "labels.json").write_text(json.dumps({
        "labels": labels,
        "input_dim": int(X.shape[1]),
        "mode": "static",
        "region": meta.get("region"),
        "hands": meta.get("hands"),
    }, indent=2))
    print(f"\nSaved model -> {out/'static_model.keras'}")
    print(f"Saved labels -> {out/'labels.json'}")
    print("Convert to TFLite (optional): "
          f"python convert_tflite.py {out/'static_model.keras'}")


if __name__ == "__main__":
    main()
