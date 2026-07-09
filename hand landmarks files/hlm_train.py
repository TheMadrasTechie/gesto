"""
Gesto — train (HandLandmarker sequences, one hand, 63 features).

Loads <label>/<n>.npy sequence samples, normalizes each frame, resamples to a
fixed length, and trains an LSTM. Saves gesto_model.h5 + labels.json.

Run:
    python hlm_train.py --data gesture_data
    python hlm_train.py --data gesture_data --frames 30
"""

import os
import json
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from hlm_common import FEATURE_DIM, normalize_sequence


def resample_sequence(seq, target_len):
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


def load_data(data_path, frames):
    classes = sorted(d for d in os.listdir(data_path)
                     if os.path.isdir(os.path.join(data_path, d)))
    if not classes:
        raise RuntimeError(f"No class folders in '{data_path}'.")

    label_map = {name: i for i, name in enumerate(classes)}
    X, y = [], []
    for name in classes:
        cdir = os.path.join(data_path, name)
        for f in os.listdir(cdir):
            if not f.endswith(".npy"):
                continue
            arr = np.load(os.path.join(cdir, f))
            if arr.ndim == 1:                       # a single frame -> (1, D)
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
                print(f"  skip {f} in '{name}': shape {arr.shape}")
                continue
            arr = normalize_sequence(arr)           # normalize each frame
            X.append(resample_sequence(arr, frames))
            y.append(label_map[name])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"Loaded {len(X)} samples, {len(classes)} classes: {classes}")
    print(f"X shape: {X.shape}")
    return X, y, label_map


def build_model(frames, n_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(frames, FEATURE_DIM)),
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
    ap.add_argument("--data", default="gesture_data")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--model", default="gesto_model.h5")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    X, y, label_map = load_data(args.data, args.frames)
    n_classes = len(label_map)
    if n_classes < 2:
        raise RuntimeError("Need at least 2 classes.")

    y_cat = to_categorical(y, num_classes=n_classes).astype(np.float32)
    counts = np.bincount(y)
    strat = y if counts.min() >= 2 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=strat)
    print(f"Train: {len(X_tr)}   Validation: {len(X_val)}")

    model = build_model(args.frames, n_classes)
    model.summary()
    early = EarlyStopping(monitor="val_categorical_accuracy", patience=40,
                          restore_best_weights=True, mode="max")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=args.epochs, batch_size=args.batch, callbacks=[early])

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {acc:.3f}")

    model.save(args.model)
    meta = {"labels": {i: n for n, i in label_map.items()},
            "frames": args.frames, "feature_dim": FEATURE_DIM}
    json.dump(meta, open(args.labels, "w"), indent=2)
    print(f"Saved model  -> {args.model}")
    print(f"Saved labels -> {args.labels}")


if __name__ == "__main__":
    main()
