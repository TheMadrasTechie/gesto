"""
Gesto two hands — STATIC train (single frame, no motion).

Trains a Dense classifier on single-frame hand poses. Use this when a gesture
is a HELD SHAPE (thumbs up, alphabet letters, most hand signs) rather than a
motion. Compared to the LSTM: far less data needed, instant detection with no
30-frame warm-up, and no collapse-to-one-class fragility.

Reads Gesto Labeller data captured with region="Hands" and hands="two",
in STATIC mode:   <project>/data/static/<label>/<uid>.npy   each of shape (126,)

Everything is open source: MediaPipe (Apache-2.0), TensorFlow (Apache-2.0),
OpenCV (Apache-2.0), NumPy (BSD).

Run:
    python train_static_hands_two.py "D:\\...\\gesto_projects\\my-project"
    python train_static_hands_two.py <project> --epochs 200 --out artifacts_static_hands_two
"""

import sys, json, argparse
from pathlib import Path
import numpy as np

REGION_KEY   = "hands_two"
GESTO_REGION = "Hands"
HANDS        = "two"
FEATURE_DIM  = 126
NORMALIZE    = True   # set False if you captured with Gesto "Normalise" unchecked

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def load(project_dir):
    """Load every static sample: X (N,63), y (N,), labels."""
    root = Path(project_dir) / "data" / "static"
    if not root.exists():
        sys.exit(
            f"No STATIC data at {root}\n"
            f"This trains on single frames. In Gesto Labeller pick the 'Static' "
            f"mode when capturing (not 'Sequence').")
    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not labels:
        sys.exit(f"No class folders under {root}")

    lmap = {n: i for i, n in enumerate(labels)}
    X, y = [], []
    for name in labels:
        for f in sorted((root / name).glob("*.npy")):
            a = np.load(f).astype(np.float32)
            a = a.reshape(-1)                     # ensure flat (63,)
            if a.shape[0] != FEATURE_DIM:
                sys.exit(f"{f} has dim {a.shape[0]} but hands_two expects "
                         f"{FEATURE_DIM}. Wrong region for this project?")
            X.append(a); y.append(lmap[name])
    if not X:
        sys.exit(f"No .npy samples found under {root}")
    return np.array(X, np.float32), np.array(y), labels


def build(n_classes, small):
    """Dense classifier for a single 126-value landmark vector.

    A static pose has no time dimension, so a small MLP is the right shape —
    no LSTM needed. Dropout keeps it honest on small datasets.
    """
    if small:
        m = Sequential([
            Input((FEATURE_DIM,)),
            Dense(64, activation="relu"), Dropout(0.3),
            Dense(32, activation="relu"), Dropout(0.2),
            Dense(n_classes, activation="softmax"),
        ])
    else:
        m = Sequential([
            Input((FEATURE_DIM,)),
            Dense(256, activation="relu"), Dropout(0.4),
            Dense(128, activation="relu"), Dropout(0.3),
            Dense(64, activation="relu"), Dropout(0.2),
            Dense(n_classes, activation="softmax"),
        ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser(description="Train static hands_two gesture model")
    ap.add_argument("project_dir", help="Gesto project folder (Copy path button)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--small", action="store_true",
                    help="Force the lighter model (auto below 300 samples)")
    ap.add_argument("--out", default="artifacts_static_hands_two")
    args = ap.parse_args()

    X, y, labels = load(args.project_dir)
    print(f"static hands_two (dim {FEATURE_DIM}): {len(X)} samples, classes={labels}")
    counts = np.bincount(y, minlength=len(labels))
    for name, c in zip(labels, counts):
        print(f"   {name:14} {c}")
    if counts.min() == 0:
        sys.exit("A class has 0 samples — capture some for every class.")
    if counts.min() < 10:
        print("NOTE: fewer than 10 samples in a class. Static models need less "
              "data than LSTMs, but aim for 20+ frames per pose for reliability.")

    # class weights so a dominant class can't win by default
    cw = {i: float(counts.sum() / (len(counts) * c)) for i, c in enumerate(counts)}
    if counts.max() >= 3 * counts.min():
        print(f"NOTE: imbalanced {dict(zip(labels, counts.tolist()))} — "
              f"applying class weights.")

    small = args.small or len(X) < 300
    if small and not args.small:
        print(f"NOTE: {len(X)} samples — using the lighter model.")

    rng = np.random.default_rng(42); idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    model = build(len(labels), small)
    model.summary()
    es = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    model.fit(X, y, validation_split=args.val_split, epochs=args.epochs,
              batch_size=args.batch_size, callbacks=[es], verbose=2,
              class_weight=cw)

    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"\nTraining-set accuracy: {acc:.3f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    (out / "labels.json").write_text(json.dumps({
        "labels": labels, "region_key": REGION_KEY, "input_dim": FEATURE_DIM,
        "mode": "static", "gesto_region": GESTO_REGION, "hands": HANDS,
        "normalized": NORMALIZE,
    }, indent=2))
    print(f"Saved -> {out/'model.keras'} and {out/'labels.json'}")


if __name__ == "__main__":
    main()
