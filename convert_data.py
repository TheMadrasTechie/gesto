"""
Gesto — convert old per-frame data into combined samples.

Source layout (one .npy per FRAME, Renotte style):
    <SOURCE>/<class>/<sequence>/<frame>.npy
e.g.
    archive/Alphabet_all_4/A/0/0.npy, 1.npy, 2.npy, ...

Output layout (one .npy per SAMPLE, Gesto style):
    gesture_data/<class>/<sample_index>.npy      # shape (frames, 63)

Each sequence folder becomes one stacked sample, frames sorted in numeric order.
If the source frames have more than 63 features (e.g. the full 1662 holistic
vector), the right-hand 63 values are sliced out so the result matches the
current 63-feature pipeline.

Run:
    python convert_data.py "archive/Alphabet_all_4"
    python convert_data.py "archive/Alphabet_all_4" --out gesture_data
"""

import os
import sys
import argparse

import numpy as np

FEATURE_DIM = 63

# In the full Renotte holistic vector (1662), the layout is:
#   pose(33*4=132) + face(468*3=1404) + left_hand(21*3=63) + right_hand(21*3=63)
# so the right hand is the LAST 63 values.
RIGHT_HAND_SLICE = slice(-63, None)


def numeric_key(name):
    stem = os.path.splitext(name)[0]
    return int(stem) if stem.isdigit() else name


def to_63(frame_vec):
    """Return a (63,) right-hand vector from a frame of any supported length."""
    v = np.asarray(frame_vec, dtype=np.float32).ravel()
    if v.shape[0] == FEATURE_DIM:
        return v
    if v.shape[0] > FEATURE_DIM:
        return v[RIGHT_HAND_SLICE].astype(np.float32)
    raise ValueError(f"frame has {v.shape[0]} features (< {FEATURE_DIM})")


def load_sequence(seq_dir):
    """Stack all frame .npy files in a sequence folder into (frames, 63)."""
    frame_files = sorted(
        (f for f in os.listdir(seq_dir) if f.endswith(".npy")),
        key=numeric_key,
    )
    frames = []
    for f in frame_files:
        arr = np.load(os.path.join(seq_dir, f))
        frames.append(to_63(arr))
    if not frames:
        return None
    return np.array(frames, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="root of the old per-frame dataset")
    ap.add_argument("--out", default="gesture_data", help="output directory")
    args = ap.parse_args()

    source = args.source
    out = args.out
    if not os.path.isdir(source):
        print(f"Source not found: {source}")
        return

    classes = sorted(
        d for d in os.listdir(source)
        if os.path.isdir(os.path.join(source, d))
    )
    if not classes:
        print("No class folders found.")
        return

    total_samples = 0
    lengths = set()

    for cls in classes:
        cls_src = os.path.join(source, cls)
        cls_out = os.path.join(out, cls)
        os.makedirs(cls_out, exist_ok=True)

        seq_dirs = sorted(
            (d for d in os.listdir(cls_src)
             if os.path.isdir(os.path.join(cls_src, d))),
            key=numeric_key,
        )

        sample_idx = 0
        for seq in seq_dirs:
            seq_path = os.path.join(cls_src, seq)
            sample = load_sequence(seq_path)
            if sample is None:
                print(f"  [{cls}] empty sequence '{seq}', skipped")
                continue
            np.save(os.path.join(cls_out, f"{sample_idx}.npy"), sample)
            lengths.add(sample.shape[0])
            sample_idx += 1

        print(f"class '{cls}': {sample_idx} samples")
        total_samples += sample_idx

    print(f"\nDone. {total_samples} samples across {len(classes)} classes "
          f"-> '{out}/'")
    if len(lengths) == 1:
        print(f"All samples have {lengths.pop()} frames.")
    else:
        print(f"WARNING: samples have varying frame counts: {sorted(lengths)}")
        print("Your fixed-frame train.py expects them all equal to NUM_FRAMES.")


if __name__ == "__main__":
    main()