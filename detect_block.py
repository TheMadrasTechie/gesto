"""
Gesto — detect by capturing 30 frames together (block mode).

Instead of a rolling window, this collects a full block of `frames` frames,
runs one prediction on that block, shows it, then starts the next block.

Uses Gesto Labeller's landmarks.py so live features exactly match training.

Requirements (same folder):
    landmarks.py
    gesto_model.h5
    labels.json     (nested: {"labels": {...}, "frames": N, "feature_dim": D})

Run:
    python detect_block.py --model gesto_model.h5 --labels labels.json

Press 'q' to quit.
"""

import json
import argparse

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from landmarks import extract_vector, normalize_vector

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DIM_TO_REGION = {
    63:  ("Hands", "one"),
    126: ("Hands", "two"),
    132: ("Pose", "two"),
    32:  ("Legs", "two"),
    258: ("Full", "two"),
}

DISPLAY_NAMES = {}


def load_meta(path):
    """Read labels.json whether it's nested or a flat {idx: name} map."""
    data = json.load(open(path))
    if isinstance(data, dict) and "labels" in data:
        labels = {int(k): v for k, v in data["labels"].items()}
        return labels, int(data.get("frames", 30)), int(data.get("feature_dim", 0))
    # flat legacy format
    labels = {int(k): v for k, v in data.items()}
    return labels, 30, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gesto_model.h5")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    model = load_model(args.model)
    index_to_name, frames, dim = load_meta(args.labels)

    # if feature_dim wasn't stored, infer from the model input
    if dim == 0:
        dim = int(model.input_shape[2])
        frames = int(model.input_shape[1])
    if dim not in DIM_TO_REGION:
        raise RuntimeError(f"Unknown feature dim {dim}; can't pick a region.")
    region, hands = DIM_TO_REGION[dim]
    print(f"Model: {frames} frames x {dim} feats -> region '{region}' ({hands})")

    def label_for(idx):
        raw = index_to_name[idx]
        return DISPLAY_NAMES.get(str(raw), raw)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return

    buffer = []
    shown_label, shown_conf = "", 0.0
    zero_vec = np.zeros(dim, dtype=np.float32)

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            mp_drawing.draw_landmarks(frame, res.left_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, res.right_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)
            if region in ("Pose", "Legs", "Full"):
                mp_drawing.draw_landmarks(frame, res.pose_landmarks,
                                          mp_holistic.POSE_CONNECTIONS)

            # extract + normalize this frame exactly like the Labeller did
            raw = extract_vector(res, region, hands)
            feat = zero_vec if raw is None else \
                normalize_vector(raw, region, hands).astype(np.float32)
            _buf = "FEAT_FULL: "
            for i in range(0, len(feat), 3):
                _buf += f"[{feat[i]:.3f},{feat[i+1]:.3f},{feat[i+2]:.3f}] "
            print(_buf)
            buffer.append(feat)

            # once a full block is collected, predict on it and reset
            if len(buffer) == frames:
                probs = model.predict(
                    np.expand_dims(np.array(buffer), 0), verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                shown_label = label_for(idx) if conf >= args.threshold else "?"
                shown_conf = conf

                ranked = sorted(
                    ((index_to_name[i], float(p)) for i, p in enumerate(probs)),
                    key=lambda kv: kv[1], reverse=True)
                allc = "  ".join(f"{DISPLAY_NAMES.get(str(n), n)}:{p:.2f}"
                                 for n, p in ranked)
                print(f"[block] {shown_label} ({conf:.2f})  |  {allc}")
                buffer = []            # start the next 30-frame block

            # header + progress bar for the current block
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (40, 24, 13), -1)
            text = f"{shown_label}  ({shown_conf:.2f})" if shown_label else "..."
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (39, 159, 239), 2, cv2.LINE_AA)
            # progress toward next prediction
            w = int(frame.shape[1] * len(buffer) / frames)
            cv2.rectangle(frame, (0, 52), (w, 58), (39, 159, 239), -1)

            cv2.imshow("Gesto - detect block (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()