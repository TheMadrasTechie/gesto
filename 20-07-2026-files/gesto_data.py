"""
gesto_data.py — shared loader for Gesto Labeller datasets.

Reads the exact on-disk layout Gesto writes:

    <project_folder>/
        project.json
        data/
            static/<label>/<uid>.npy      # each file shape (D,)
            sequence/<label>/<uid>.npy     # each file shape (T, D)

Feature dimension D depends on the project's region:
    Hands (one hand)  -> 63     (21 landmarks x 3)
    Hands (two hands) -> 126    (2 x 21 x 3)
    Pose              -> 132    (33 x 4: x,y,z,visibility)
    Legs              -> 32     (8 x 4)
    Full              -> 258    (pose 132 + hands 126)

Point these scripts at a project folder (the "Copy path" button on a project
card gives you exactly this path).
"""

import json
from pathlib import Path
import numpy as np


def load_project_meta(project_dir):
    """Return the project.json dict (region, hands, classes, ...)."""
    p = Path(project_dir) / "project.json"
    if not p.exists():
        raise FileNotFoundError(
            f"No project.json in {project_dir}. Point this at a Gesto project "
            f"folder (use the 'Copy path' button on a project card).")
    return json.loads(p.read_text(encoding="utf-8"))


def load_static(project_dir):
    """Load all single-frame samples.

    Returns (X, y, labels):
        X      float32 array, shape (N, D)
        y      int array, shape (N,)   -- class index into `labels`
        labels list[str]               -- class names, index == label id
    """
    root = Path(project_dir) / "data" / "static"
    return _load_dir(root, sequence=False)


def load_sequence(project_dir, pad_to=None):
    """Load all motion samples.

    Sequences can have different lengths (T varies). By default they're padded
    (post-padding with zeros) / truncated to the longest sequence found, so the
    result is a single dense array an LSTM can batch. Pass pad_to=N to force a
    fixed length.

    Returns (X, y, labels, seq_len):
        X       float32 array, shape (N, T, D)
        y       int array, shape (N,)
        labels  list[str]
        seq_len int  -- the T every sequence was padded/truncated to
    """
    root = Path(project_dir) / "data" / "sequence"
    seqs, y, labels = _load_dir(root, sequence=True)
    if len(seqs) == 0:
        return np.empty((0, 0, 0), np.float32), np.empty((0,), int), labels, 0

    dim = seqs[0].shape[-1]
    T = pad_to or max(s.shape[0] for s in seqs)
    X = np.zeros((len(seqs), T, dim), np.float32)
    for i, s in enumerate(seqs):
        n = min(s.shape[0], T)
        X[i, :n] = s[:n]
    return X, np.asarray(y, int), labels, T


def _load_dir(root, sequence):
    """Walk <root>/<label>/*.npy and collect arrays + integer labels."""
    if not root.exists():
        kind = "sequence" if sequence else "static"
        raise FileNotFoundError(
            f"No {kind} data at {root}. Capture some {kind} annotations first.")

    labels = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not labels:
        raise ValueError(f"No class folders under {root}.")

    label_to_idx = {name: i for i, name in enumerate(labels)}
    X, y = [], []
    for name in labels:
        for f in sorted((root / name).glob("*.npy")):
            arr = np.load(f).astype(np.float32)
            if sequence and arr.ndim == 1:      # a 1-frame sequence saved as (D,)
                arr = arr[None, :]
            X.append(arr)
            y.append(label_to_idx[name])

    if not sequence:
        X = np.asarray(X, np.float32)           # (N, D) — all same length
    return X, y, labels    # for sequence, X stays a list (ragged) until padded


if __name__ == "__main__":
    # quick sanity check: python gesto_data.py <project_folder>
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    meta = load_project_meta(d)
    print(f"Project: {meta.get('name')}  region={meta.get('region')} "
          f"hands={meta.get('hands')}")
    print("Classes:", meta.get("classes"))
    try:
        Xs, ys, ls = load_static(d)
        print(f"Static  : {len(Xs)} samples, dim={Xs.shape[1] if len(Xs) else '-'}, "
              f"classes={ls}")
    except (FileNotFoundError, ValueError) as e:
        print("Static  :", e)
    try:
        Xq, yq, lq, T = load_sequence(d)
        print(f"Sequence: {len(Xq)} samples, shape={Xq.shape if len(Xq) else '-'}, "
              f"padded T={T}, classes={lq}")
    except (FileNotFoundError, ValueError) as e:
        print("Sequence:", e)
