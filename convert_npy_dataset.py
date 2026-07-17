"""
convert_npy_dataset.py

Converts a .npy-based gesture dataset (the kind captured with older
Holistic-based tooling: data/<label>/<index>.npy, each array shaped
(NUM_FRAMES, 63)) into the JSON format train_gesture.py already expects:

    data/<label>/<index>.json
    {"label": ..., "num_frames": 30, "num_landmarks": 21, "landmarks": [...]}

This is a pure RESHAPE/FORMAT conversion -- it does NOT normalize the data.
train_gesture.py does its own wrist-relative normalization at load time, so
passing it raw landmarks (as this script does) is correct; don't
double-normalize here.

REMINDER (same caveat as before): if this data was captured via MediaPipe
Holistic and you plan to train a model that runs live inference through the
Flutter app's Tasks HandLandmarker-based capture/detect screens, the two
engines have measured, real disagreement (see our earlier comparison) --
particularly in the z-axis and fingertip precision. Safest use of this
converted data: train and evaluate within a consistent Holistic-only
pipeline, OR treat it as a separate experiment from your live Flutter model
until you've validated cross-engine agreement is close enough for your needs.

Usage:
    python convert_npy_dataset.py --input gesture_data --output data

    (--input: folder containing <label>/<index>.npy)
    (--output: destination folder, defaults to "data" -- the same folder
     train_gesture.py reads from)
"""

import argparse
import glob
import json
import os

import numpy as np

NUM_FRAMES = 30
NUM_LANDMARKS = 21
NUM_COORDS = 3


def convert_file(npy_path: str, label: str, out_dir: str, index: str):
    arr = np.load(npy_path)

    if arr.shape != (NUM_FRAMES, NUM_LANDMARKS * NUM_COORDS):
        print(f"  skipping {npy_path}: unexpected shape {arr.shape} "
              f"(expected ({NUM_FRAMES}, {NUM_LANDMARKS * NUM_COORDS}))")
        return False

    reshaped = arr.reshape(NUM_FRAMES, NUM_LANDMARKS, NUM_COORDS)

    payload = {
        "label": label,
        "num_frames": NUM_FRAMES,
        "num_landmarks": NUM_LANDMARKS,
        "landmarks": reshaped.tolist(),
    }

    label_out_dir = os.path.join(out_dir, label)
    os.makedirs(label_out_dir, exist_ok=True)
    out_path = os.path.join(label_out_dir, f"{index}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="gesture_data",
                         help="folder containing <label>/<index>.npy files")
    parser.add_argument("--output", default="data",
                         help="destination folder (train_gesture.py reads from 'data' by default)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input folder '{args.input}' not found.")

    labels = sorted(
        d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))
    )
    if not labels:
        raise ValueError(f"No label subfolders found under '{args.input}'.")

    total_converted = 0
    total_skipped = 0

    for label in labels:
        npy_files = glob.glob(os.path.join(args.input, label, "*.npy"))
        converted_for_label = 0
        for npy_path in npy_files:
            index = os.path.splitext(os.path.basename(npy_path))[0]
            ok = convert_file(npy_path, label, args.output, index)
            if ok:
                converted_for_label += 1
                total_converted += 1
            else:
                total_skipped += 1
        print(f"{label}: converted {converted_for_label}/{len(npy_files)} samples")

    print(f"\nDone. {total_converted} samples converted, {total_skipped} skipped.")
    print(f"Output written to '{args.output}/' -- ready for train_gesture.py "
          f"(point its DATA_DIR at '{args.output}' if it's not already 'data').")


if __name__ == "__main__":
    main()
