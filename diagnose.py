"""
diagnose.py — inspect a Gesto project's sequence data before training.

Tells you WHY detection might collapse to one class:
- how many sequences per class (imbalance)
- how many would be SKIPPED at a given --seq_len (too-short clips)
- the length distribution of your clips (consistency)
- the feature dimension actually on disk (region match)

Usage:
    python diagnose.py <project_dir> --seq_len 30
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("--seq_len", type=int, default=30)
    args = ap.parse_args()

    root = Path(args.project_dir) / "data" / "sequence"
    if not root.exists():
        raise SystemExit(f"No sequence data at {root}\n"
                         f"(Did you capture Static instead of Sequence? "
                         f"Check {Path(args.project_dir)/'data'})")

    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    print(f"Project: {Path(args.project_dir).name}")
    print(f"Classes found: {labels}\n")

    dims_seen = set()
    total_usable = 0
    grand_lengths = []
    print(f"{'class':16} {'clips':>6} {'usable(>= '+str(args.seq_len)+')':>16} "
          f"{'min':>4} {'max':>4} {'median':>7}")
    print("-" * 60)
    per_class_usable = {}
    for name in labels:
        files = sorted((root / name).glob("*.npy"))
        lengths = []
        usable = 0
        for f in files:
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr[None, :]
            lengths.append(arr.shape[0])
            dims_seen.add(arr.shape[1])
            if arr.shape[0] >= args.seq_len:
                usable += 1
        per_class_usable[name] = usable
        total_usable += usable
        grand_lengths += lengths
        if lengths:
            print(f"{name:16} {len(files):>6} {usable:>16} "
                  f"{min(lengths):>4} {max(lengths):>4} "
                  f"{int(np.median(lengths)):>7}")
        else:
            print(f"{name:16} {0:>6} {0:>16}    -    -       -")

    print("-" * 60)
    print(f"\nFeature dimension(s) on disk: {sorted(dims_seen)}")
    if len(dims_seen) > 1:
        print("  WARNING: mixed dimensions! Some samples were captured with a "
              "different region/hands setting. Recapture consistently.")

    print(f"Total usable sequences at seq_len={args.seq_len}: {total_usable}")

    # ---- verdicts ----
    print("\n=== Likely diagnosis ===")
    problems = False
    empty = [n for n, u in per_class_usable.items() if u == 0]
    if empty:
        problems = True
        print(f"* These classes have ZERO usable sequences: {empty}")
        print(f"  -> at seq_len={args.seq_len} their clips are all too short.")
        if grand_lengths:
            print(f"  -> try --seq_len {int(np.median(grand_lengths))} "
                  f"(your median clip length), or recapture longer clips.")
    counts = list(per_class_usable.values())
    if counts and max(counts) > 0:
        imbalance = max(counts) / max(1, min(c for c in counts if c >= 0))
        biggest = max(per_class_usable, key=per_class_usable.get)
        if min(counts) == 0 or (max(counts) >= 3 * max(1, min(counts))):
            problems = True
            print(f"* Class imbalance: '{biggest}' dominates "
                  f"({per_class_usable}).")
            print("  -> the model can collapse to always predicting the "
                  "majority class. Capture a similar count per class.")
    if total_usable < 15 * max(1, len(labels)):
        problems = True
        print(f"* Low data: {total_usable} usable sequences across "
              f"{len(labels)} classes.")
        print("  -> aim for at least ~15-30 sequences PER class.")
    if grand_lengths:
        lo, hi = min(grand_lengths), max(grand_lengths)
        if hi - lo > max(10, args.seq_len // 2):
            print(f"* Inconsistent clip length (min={lo}, max={hi}).")
            print("  -> set 'Max frames' in Gesto so every capture is the "
                  "same length; trains far more reliably.")
    if not problems:
        print("Data looks balanced and sufficient. If detection still "
              "collapses, train longer (--epochs 400) or lower --threshold.")


if __name__ == "__main__":
    main()