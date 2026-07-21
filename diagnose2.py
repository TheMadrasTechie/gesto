"""
diagnose2.py — deeper check for WHY an LSTM collapses to one class.

Beyond counts, this checks:
1. MOTION: do your sequences actually move over time? (LSTMs need motion; a
   held pose gives an LSTM nothing to learn -> collapse.)
2. SEPARABILITY: are the classes even distinguishable in the raw features?
   (If class means overlap heavily, no model can separate them.)
3. VALUE SANITY: are the landmark values in a sane 0..1-ish range, not all
   zeros (undetected hands)?

Usage:
    python diagnose2.py <project_dir> --seq_len 30
"""
import argparse
from pathlib import Path
import numpy as np


def load_class_arrays(root, seq_len):
    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    data = {}
    for name in labels:
        seqs = []
        for f in sorted((root / name).glob("*.npy")):
            a = np.load(f).astype(np.float32)
            if a.ndim == 1:
                a = a[None, :]
            if a.shape[0] >= seq_len:
                seqs.append(a[:seq_len])
        if seqs:
            data[name] = np.stack(seqs)   # (N, T, D)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("--seq_len", type=int, default=30)
    args = ap.parse_args()

    root = Path(args.project_dir) / "data" / "sequence"
    data = load_class_arrays(root, args.seq_len)
    if not data:
        raise SystemExit("No usable sequences. Run diagnose.py first.")

    labels = list(data)
    print(f"Classes with usable data: {labels}\n")

    # ---- 1. motion check: mean frame-to-frame change within a sequence ----
    print("=== MOTION (frame-to-frame change within each gesture) ===")
    print("Low motion + LSTM = likely collapse. These may need the STATIC model.")
    low_motion = True
    for name, arr in data.items():
        # mean abs diff between consecutive frames, averaged over samples & dims
        diffs = np.abs(np.diff(arr, axis=1))          # (N, T-1, D)
        motion = float(diffs.mean())
        nonzero = float((np.abs(arr) > 1e-6).mean())  # fraction of non-zero values
        flag = "  <-- almost still" if motion < 0.005 else ""
        if motion >= 0.005:
            low_motion = False
        print(f"  {name:14} motion={motion:.4f}  nonzero_values={nonzero*100:4.0f}%{flag}")
    if low_motion:
        print("\n  VERDICT: very little motion in any class. An LSTM has nothing")
        print("  temporal to learn -> it will collapse. Use the STATIC model")
        print("  (train_static.py) or capture gestures that actually move.")

    # ---- 2. separability: distance between class means vs within-class spread
    print("\n=== SEPARABILITY (can the classes even be told apart?) ===")
    means = {n: arr.reshape(len(arr), -1).mean(0) for n, arr in data.items()}
    flat = {n: arr.reshape(len(arr), -1) for n, arr in data.items()}
    within = np.mean([flat[n].std(0).mean() for n in labels])
    between = []
    names = list(labels)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            between.append(np.linalg.norm(means[names[i]] - means[names[j]]))
    between_mean = float(np.mean(between)) if between else 0.0
    print(f"  avg distance BETWEEN class means : {between_mean:.3f}")
    print(f"  avg spread WITHIN classes        : {within:.3f}")
    ratio = between_mean / (within + 1e-9)
    print(f"  separability ratio (higher=better): {ratio:.2f}")
    if ratio < 1.0:
        print("  VERDICT: classes overlap heavily in feature space. The gestures")
        print("  look too similar to the model. Make them more distinct, capture")
        print("  more consistently, or check they were labelled correctly.")
    else:
        print("  Classes look separable — data is probably fine.")

    # ---- 3. all-zero check (undetected hands) ----
    print("\n=== DETECTION QUALITY (all-zero frames = hand not detected) ===")
    for name, arr in data.items():
        zero_frac = float((np.abs(arr).sum(axis=2) < 1e-6).mean())
        flag = "  <-- many empty frames!" if zero_frac > 0.3 else ""
        print(f"  {name:14} empty_frames={zero_frac*100:4.0f}%{flag}")


if __name__ == "__main__":
    main()