"""
train_lstm.py — train a sequence (motion) gesture classifier with an LSTM.

Reads Gesto's sequence/ .npy samples (each is a (T, D) landmark sequence) and
trains a stacked-LSTM classifier. This is the right model when a gesture is a
MOTION over time (waving, clapping, walking) rather than a single held pose.

Architecture matches the sign-language project's proven style: 3 LSTM layers +
Dense head. Variable-length sequences are zero-padded to a common length
(saved in labels.json so detection uses the same T).

Outputs (into --out, default ./artifacts_lstm):
    lstm_model.keras       the trained Keras model
    labels.json            classes + feature dim + sequence length T
    (optional) lstm_model.tflite via convert_tflite.py

Usage:
    python train_lstm.py /path/to/project_folder
    python train_lstm.py /path/to/project_folder --seq_len 30 --epochs 120

Note on TFLite: plain conversion of LSTM layers can silently produce wrong
numbers. Use the accompanying convert_tflite.py (forces unroll=True + verifies),
which is the fix already validated on the sign-language project.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gesto_data import load_sequence, load_project_meta


def build_model(seq_len, input_dim, num_classes):
    """Stacked LSTM classifier (3 LSTM + Dense head).

    unroll=False here for training speed; convert_tflite.py re-clones with
    unroll=True at conversion time (that's the combination that converts
    correctly while keeping training fast).
    """
    model = keras.Sequential([
        keras.Input(shape=(seq_len, input_dim)),
        layers.LSTM(64, return_sequences=True, activation="tanh"),
        layers.LSTM(128, return_sequences=True, activation="tanh"),
        layers.LSTM(64, return_sequences=False, activation="tanh"),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir", help="Path to a Gesto project folder")
    ap.add_argument("--seq_len", type=int, default=None,
                    help="Fixed sequence length T (default: longest in dataset)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--out", default="artifacts_lstm")
    args = ap.parse_args()

    meta = load_project_meta(args.project_dir)
    X, y, labels, T = load_sequence(args.project_dir, pad_to=args.seq_len)
    if len(X) == 0:
        raise SystemExit("No sequence samples found. Capture some motion first.")

    print(f"Project '{meta.get('name')}'  region={meta.get('region')}")
    print(f"Loaded {len(X)} sequences, shape={X.shape} (T={T}, dim={X.shape[2]}), "
          f"classes={labels}")
    counts = np.bincount(y, minlength=len(labels))
    for name, c in zip(labels, counts):
        print(f"   {name:20} {c} sequences")
    if counts.min() < 5:
        print("WARNING: some classes have <5 sequences; expect unreliable accuracy.")
    # sequence-length variance directly hurts live detection: if clips vary a
    # lot in length, most training samples become mostly zero-padding while live
    # detection feeds full windows. Flag it so you can capture more consistently.
    raw_lens = [int((X[i].any(axis=1)).sum()) for i in range(len(X))]
    if raw_lens:
        lo, hi = min(raw_lens), max(raw_lens)
        print(f"Sequence lengths: min={lo}, max={hi}, T(pad)={T}")
        if hi - lo > max(8, T // 2):
            print("WARNING: large variation in clip length. For steadier live "
                  "detection, try to record each gesture at a similar duration, "
                  "or pass a fixed --seq_len close to your typical clip length.")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    model = build_model(T, X.shape[2], len(labels))
    model.summary()

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=18, restore_best_weights=True)
    model.fit(X, y, validation_split=args.val_split,
              epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[es], verbose=2)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save(out / "lstm_model.keras")
    (out / "labels.json").write_text(json.dumps({
        "labels": labels,
        "input_dim": int(X.shape[2]),
        "seq_len": int(T),
        "mode": "sequence",
        "region": meta.get("region"),
        "hands": meta.get("hands"),
    }, indent=2))
    print(f"\nSaved model -> {out/'lstm_model.keras'}")
    print(f"Saved labels -> {out/'labels.json'}")
    print("Convert to TFLite (recommended): "
          f"python convert_tflite.py {out/'lstm_model.keras'}")


if __name__ == "__main__":
    main()