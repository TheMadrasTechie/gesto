"""
train.py — LSTM trainer matching your Alphabet_recognition notebook, for any
region. Fixed-length sequences (no padding), your exact architecture.

Reads Gesto sequence data:  <project>/data/sequence/<label>/<uid>.npy  (each (T,D))
Only sequences with exactly --seq_len frames are used (matching the notebook,
where every video was exactly 30 frames). Longer ones are trimmed to the first
--seq_len; shorter ones are skipped with a warning.

Usage:
    python train.py <project_dir> --region hands_right
    python train.py <project_dir> --region pose --seq_len 30 --epochs 300

Regions: hands_right (63), hands_two (126), pose (132), legs (32), full (258)
"""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from gesto_regions import REGIONS


def load(project_dir, dim, seq_len):
    root = Path(project_dir) / "data" / "sequence"
    if not root.exists():
        raise SystemExit(f"No sequence data at {root}")
    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not labels:
        raise SystemExit(f"No class folders under {root}")
    lmap = {n: i for i, n in enumerate(labels)}
    X, y, skipped = [], [], 0
    for name in labels:
        for f in sorted((root / name).glob("*.npy")):
            arr = np.load(f).astype(np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[1] != dim:
                raise SystemExit(
                    f"{f} has dim {arr.shape[1]} but region expects {dim}. "
                    f"Wrong --region for this project?")
            if arr.shape[0] < seq_len:
                skipped += 1
                continue
            X.append(arr[:seq_len])              # trim to exactly seq_len
            y.append(lmap[name])
    if skipped:
        print(f"NOTE: skipped {skipped} sequences shorter than {seq_len} frames.")
    if not X:
        raise SystemExit(f"No sequences with >= {seq_len} frames found.")
    return np.array(X, np.float32), np.array(y), labels


def build(seq_len, dim, n):
    m = Sequential([
        Input(shape=(seq_len, dim)),
        LSTM(64, return_sequences=True, activation="relu"),
        LSTM(128, return_sequences=True, activation="relu"),
        LSTM(64, return_sequences=False, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(n, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("--region", required=True, choices=list(REGIONS))
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = REGIONS[args.region]
    out = Path(args.out or f"artifacts_{args.region}")
    X, y, labels = load(args.project_dir, spec["dim"], args.seq_len)
    print(f"Region {args.region} (dim {spec['dim']}): {len(X)} sequences, "
          f"classes={labels}")
    for name, c in zip(labels, np.bincount(y, minlength=len(labels))):
        print(f"   {name:12} {c}")

    rng = np.random.default_rng(42); idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    model = build(args.seq_len, spec["dim"], len(labels))
    model.summary()
    es = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    model.fit(X, y, validation_split=args.val_split, epochs=args.epochs,
              batch_size=args.batch_size, callbacks=[es], verbose=2)

    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    (out / "labels.json").write_text(json.dumps({
        "labels": labels, "region_key": args.region, "input_dim": spec["dim"],
        "seq_len": args.seq_len,
        "gesto_region": spec["gesto_region"], "hands": spec["hands"],
    }, indent=2))
    print(f"\nSaved -> {out/'model.keras'} and {out/'labels.json'}")


if __name__ == "__main__":
    main()
