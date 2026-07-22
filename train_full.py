"""
Gesto full body + hands — train.

Self-contained train script for the full region (dim 258).
Reads Gesto Labeller data captured with region="Full".
No other project files needed — just this file.

Matches Gesto's capture exactly: MediaPipe Holistic, the same landmark ordering,
the same normalization (Gesto's "Normalise" is ON by default), and the webcam
mirror. If you captured with Normalise UNCHECKED, set NORMALIZE = False below.
"""

import os, sys, json, argparse
from pathlib import Path
from collections import deque
import numpy as np

REGION_KEY   = "full"
GESTO_REGION = "Full"
HANDS        = "two"
FEATURE_DIM  = 258
NORMALIZE    = True   # set False if you captured with Gesto "Normalise" unchecked

LEG_POSE_IDX = [23, 24, 25, 26, 27, 28, 31, 32]


# ---------- landmark extraction (matches Gesto capture) ----------
def _rh(res):
    return (np.array([[r.x, r.y, r.z] for r in res.right_hand_landmarks.landmark]).flatten()
            if res.right_hand_landmarks else np.zeros(63, np.float32))

def _lh(res):
    return (np.array([[r.x, r.y, r.z] for r in res.left_hand_landmarks.landmark]).flatten()
            if res.left_hand_landmarks else np.zeros(63, np.float32))

def _pose_full(res):
    return (np.array([[r.x, r.y, r.z, r.visibility] for r in res.pose_landmarks.landmark]).flatten()
            if res.pose_landmarks else np.zeros(132, np.float32))

def _pose_legs(res):
    if not res.pose_landmarks:
        return np.zeros(len(LEG_POSE_IDX) * 4, np.float32)
    lm = res.pose_landmarks.landmark
    return np.array([[lm[i].x, lm[i].y, lm[i].z, lm[i].visibility] for i in LEG_POSE_IDX]).flatten()

def extract(res):
    """Return the full feature vector (raw, unnormalized)."""
    return np.concatenate([_pose_full(res), _lh(res), _rh(res)]).astype(np.float32)


# ---------- normalization (EXACT copy of Gesto's normalize_vector) ----------
def _norm_one_hand(pts):
    if np.any(pts):
        pts = pts - pts[0]
        s = np.linalg.norm(pts, axis=1).max()
        if s > 1e-6:
            pts = pts / s
    return pts

def normalize(vec):
    vec = np.asarray(vec, np.float32)
    if REGION_KEY == "hands_one":
        return _norm_one_hand(vec.reshape(21, 3).copy()).reshape(-1)
    if REGION_KEY == "hands_two":
        pts = vec.reshape(2, 21, 3).copy()
        for h in range(2):
            pts[h] = _norm_one_hand(pts[h])
        return pts.reshape(-1)
    if REGION_KEY in ("pose", "legs"):
        n = 33 if REGION_KEY == "pose" else len(LEG_POSE_IDX)
        pts = vec.reshape(n, 4).copy()
        xyz = pts[:, :3]
        mask = np.any(xyz != 0, axis=1)
        if np.any(mask):
            xyz -= xyz[mask].mean(axis=0)
            s = np.linalg.norm(xyz, axis=1).max()
            if s > 1e-6:
                xyz /= s
        pts[:, :3] = xyz
        return pts.reshape(-1)
    if REGION_KEY == "full":
        # pose(132) normalized as pose, hands(126) normalized per-hand
        pose = vec[:132].reshape(33, 4).copy()
        xyz = pose[:, :3]; mask = np.any(xyz != 0, axis=1)
        if np.any(mask):
            xyz -= xyz[mask].mean(axis=0)
            s = np.linalg.norm(xyz, axis=1).max()
            if s > 1e-6: xyz /= s
        pose[:, :3] = xyz
        hands = vec[132:].reshape(2, 21, 3).copy()
        for h in range(2):
            hands[h] = _norm_one_hand(hands[h])
        return np.concatenate([pose.reshape(-1), hands.reshape(-1)])
    return vec


# ---------- drawing (region-aware) ----------
_HAND_BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
               (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),
               (18,19),(19,20),(0,17)]
_POSE_BONES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
               (23,25),(25,27),(27,29),(27,31),(24,26),(26,28),(28,30),(28,32)]
_LEG_BONES  = [(0,2),(2,4),(4,6),(1,3),(3,5),(5,7),(0,1)]

def _dots(frame, pts, bones):
    import cv2
    h, w = frame.shape[:2]
    for a, b in bones:
        if a < len(pts) and b < len(pts):
            ax, ay = pts[a]; bx, by = pts[b]
            if (ax or ay) and (bx or by):
                cv2.line(frame, (int(ax*w),int(ay*h)), (int(bx*w),int(by*h)), (60,180,75), 2)
    for x, y in pts:
        if x or y:
            cv2.circle(frame, (int(x*w),int(y*h)), 3, (40,160,230), -1)

def draw(frame, res):
    hx = lambda h: [(lm.x, lm.y) for lm in h.landmark] if h else []
    if REGION_KEY == "hands_one":
        hand = res.right_hand_landmarks or res.left_hand_landmarks
        if hand: _dots(frame, hx(hand), _HAND_BONES)
    elif REGION_KEY == "hands_two":
        for h in (res.left_hand_landmarks, res.right_hand_landmarks):
            if h: _dots(frame, hx(h), _HAND_BONES)
    elif REGION_KEY == "pose" and res.pose_landmarks:
        _dots(frame, [(lm.x,lm.y) for lm in res.pose_landmarks.landmark], _POSE_BONES)
    elif REGION_KEY == "legs" and res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        _dots(frame, [(lm[i].x, lm[i].y) for i in LEG_POSE_IDX], _LEG_BONES)
    elif REGION_KEY == "full":
        if res.pose_landmarks:
            _dots(frame, [(lm.x,lm.y) for lm in res.pose_landmarks.landmark], _POSE_BONES)
        for h in (res.left_hand_landmarks, res.right_hand_landmarks):
            if h: _dots(frame, hx(h), _HAND_BONES)

def make_holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)


# ---------- training ----------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def load(project_dir, seq_len):
    root = Path(project_dir) / "data" / "sequence"
    if not root.exists():
        sys.exit(f"No sequence data at {root}")
    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not labels:
        sys.exit(f"No class folders under {root}")
    lmap = {n: i for i, n in enumerate(labels)}
    X, y, skipped = [], [], 0
    for name in labels:
        for f in sorted((root / name).glob("*.npy")):
            a = np.load(f).astype(np.float32)
            if a.ndim == 1:
                a = a[None, :]
            if a.shape[1] != FEATURE_DIM:
                sys.exit(f"{f} has dim {a.shape[1]} but {REGION_KEY} expects "
                         f"{FEATURE_DIM}. Wrong region for this project?")
            if a.shape[0] < seq_len:
                skipped += 1; continue
            X.append(a[:seq_len]); y.append(lmap[name])
    if skipped:
        print(f"NOTE: skipped {skipped} sequences shorter than {seq_len} frames.")
    if not X:
        sys.exit(f"No sequences with >= {seq_len} frames.")
    return np.array(X, np.float32), np.array(y), labels


def build(seq_len, n, small):
    if small:
        m = Sequential([Input((seq_len, FEATURE_DIM)),
                        LSTM(32, activation="tanh"), Dropout(0.4),
                        Dense(32, activation="relu"), Dropout(0.3),
                        Dense(n, activation="softmax")])
    else:
        m = Sequential([Input((seq_len, FEATURE_DIM)),
                        LSTM(64, return_sequences=True, activation="tanh"),
                        LSTM(128, return_sequences=True, activation="tanh"),
                        LSTM(64, activation="tanh"),
                        Dense(64, activation="relu"), Dense(32, activation="relu"),
                        Dense(n, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser(description="Train full gesture model")
    ap.add_argument("project_dir", help="Gesto project folder (Copy path button)")
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--out", default="artifacts_full")
    args = ap.parse_args()

    X, y, labels = load(args.project_dir, args.seq_len)
    print(f"full (dim 258): {len(X)} sequences, classes={labels}")
    counts = np.bincount(y, minlength=len(labels))
    for name, c in zip(labels, counts):
        print(f"   {name:14} {c}")
    if counts.min() == 0:
        sys.exit("A class has 0 usable sequences (clips too short). "
                 "Lower --seq_len or capture longer clips.")

    small = args.small or len(X) < 100
    if small and not args.small:
        print(f"NOTE: {len(X)} sequences — using lighter model (avoids collapse "
              f"on small data).")
    cw = {i: float(counts.sum() / (len(counts) * c)) for i, c in enumerate(counts)}

    rng = np.random.default_rng(42); idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    model = build(args.seq_len, len(labels), small)
    model.summary()
    es = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    model.fit(X, y, validation_split=0.15, epochs=args.epochs,
              batch_size=args.batch_size, callbacks=[es], verbose=2, class_weight=cw)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    (out / "labels.json").write_text(json.dumps({
        "labels": labels, "region_key": REGION_KEY, "input_dim": FEATURE_DIM,
        "seq_len": args.seq_len, "gesto_region": GESTO_REGION, "hands": HANDS,
        "normalized": NORMALIZE,
    }, indent=2))
    print(f"\nSaved -> {out/'model.keras'} and {out/'labels.json'}")


if __name__ == "__main__":
    main()
