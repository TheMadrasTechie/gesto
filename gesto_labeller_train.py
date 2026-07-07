"""
Gesto — train on Gesto Labeller workspace data.

Reads a Gesto Labeller project workspace directly:

    <workspace>/<project>/
        project.json
        manifest.csv                       # uid,label,mode,frames,dim,path
        data/
            sequence/<label>/<uid>.npy     # (T, D)
            static/<label>/<uid>.npy       # (D,)

Trains an LSTM on the SEQUENCE samples and saves:
    gesto_model.h5
    labels.json      (index -> label name)

Feature dimension D is detected automatically from the data (Hands 126,
Pose 132, Legs 32, Full 258), so this works for any region.

Run:
    python gesto_labeller_train.py --project "C:/Users/you/GestoStudio/my-project"
    python gesto_labeller_train.py --project "..." --frames 30
    python gesto_labeller_train.py --project "..." --model asl_model.h5
"""

import os
import csv
import json
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# ----------------------- data loading -----------------------

def resample_sequence(seq, target_len):
    """Stretch/compress a (L, D) sequence to (target_len, D) by interpolation."""
    seq = np.asarray(seq, dtype=np.float32)
    L, D = seq.shape
    if L == target_len:
        return seq
    if L == 1:
        return np.repeat(seq, target_len, axis=0)
    old = np.linspace(0.0, 1.0, L)
    new = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(new, old, seq[:, j]) for j in range(D)],
                    axis=1).astype(np.float32)


def _coerce_sample(arr):
    """Return a (L, D) sequence from either a (D,) static or (T, D) sequence array."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:              # static: one frame -> (1, D)
        return arr.reshape(1, -1)
    if arr.ndim == 2:              # sequence: (T, D)
        return arr
    return None


def load_from_manifest(project_dir):
    """Load static AND sequence samples using manifest.csv.
    Returns (samples, labels, dim)."""
    manifest = os.path.join(project_dir, "manifest.csv")
    samples, labels, dims = [], [], set()

    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            path = row.get("path", "")
            if not path:
                continue
            if not os.path.isabs(path):
                path = os.path.join(project_dir, path)
            if not os.path.exists(path):
                print(f"  missing file, skipped: {path}")
                continue
            arr = np.load(path)
            seq = _coerce_sample(arr)
            if seq is None:
                print(f"  odd shape {arr.shape}, skipped: {path}")
                continue
            samples.append(seq)
            labels.append(row["label"])
            dims.add(seq.shape[1])

    if len(dims) > 1:
        raise RuntimeError(f"Mixed feature dims in data: {sorted(dims)}. "
                           f"All samples must be the same region.")
    dim = dims.pop() if dims else 0
    return samples, labels, dim


def load_from_folders(project_dir):
    """Fallback: scan data/sequence/ AND data/static/ folders if no manifest."""
    samples, labels, dims = [], [], set()
    for mode in ("sequence", "static"):
        root = os.path.join(project_dir, "data", mode)
        if not os.path.isdir(root):
            continue
        for label in sorted(os.listdir(root)):
            ldir = os.path.join(root, label)
            if not os.path.isdir(ldir):
                continue
            for f in sorted(os.listdir(ldir)):
                if not f.endswith(".npy"):
                    continue
                seq = _coerce_sample(np.load(os.path.join(ldir, f)))
                if seq is None:
                    continue
                samples.append(seq)
                labels.append(label)
                dims.add(seq.shape[1])
    if len(dims) > 1:
        raise RuntimeError(f"Mixed feature dims: {sorted(dims)}.")
    dim = dims.pop() if dims else 0
    return samples, labels, dim


def load_data(project_dir, frames):
    if os.path.exists(os.path.join(project_dir, "manifest.csv")):
        samples, labels, dim = load_from_manifest(project_dir)
    else:
        print("No manifest.csv — scanning data/sequence/ folders instead.")
        samples, labels, dim = load_from_folders(project_dir)

    if not samples:
        raise RuntimeError(
            "No samples found in this project (checked manifest + "
            "data/sequence and data/static)."
        )

    # resample every sample to a common length so shapes match
    X = np.array([resample_sequence(s, frames) for s in samples],
                 dtype=np.float32)

    class_names = sorted(set(labels))
    label_map = {name: i for i, name in enumerate(class_names)}
    y = np.array([label_map[l] for l in labels])

    print(f"Loaded {len(X)} sequence samples across {len(class_names)} classes.")
    print(f"Classes: {class_names}")
    print(f"Feature dim (region): {dim}   |   frames: {frames}")
    print(f"X shape: {X.shape}")
    return X, y, label_map, dim


# ----------------------- model -----------------------

def build_model(frames, feature_dim, n_classes):
    model = Sequential([
        # Masking lets padded/zero frames be ignored if present.
        Masking(mask_value=0.0, input_shape=(frames, feature_dim)),
        LSTM(64, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation="tanh"),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True,
                    help="path to a Gesto Labeller project folder")
    ap.add_argument("--frames", type=int, default=30,
                    help="fixed sequence length to resample to (default 30)")
    ap.add_argument("--model", default="gesto_model.h5")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    X, y, label_map, dim = load_data(args.project, args.frames)
    n_classes = len(label_map)
    if n_classes < 2:
        raise RuntimeError("Need at least 2 classes to train.")

    y_cat = to_categorical(y, num_classes=n_classes).astype(np.float32)

    # stratify only if every class has >= 2 samples
    counts = np.bincount(y)
    strat = y if counts.min() >= 2 else None
    if strat is None:
        print("Note: some class has <2 samples; splitting without stratify.")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=strat
    )
    print(f"Train: {len(X_tr)}   Validation: {len(X_val)}")

    model = build_model(args.frames, dim, n_classes)
    model.summary()

    early = EarlyStopping(monitor="val_categorical_accuracy", patience=40,
                          restore_best_weights=True, mode="max")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=args.epochs, batch_size=args.batch, callbacks=[early])

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {acc:.3f}")

    model.save(args.model)
    index_to_name = {i: name for name, i in label_map.items()}
    # store frames + dim so detect can configure itself
    meta = {"labels": index_to_name, "frames": args.frames, "feature_dim": dim}
    with open(args.labels, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model  -> {args.model}")
    print(f"Saved labels -> {args.labels}")


if __name__ == "__main__":
    main()