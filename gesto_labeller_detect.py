"""
Gesto — detect for Gesto Labeller models (uses the Labeller's own landmarks.py).

This imports Gesto Labeller's extraction + normalization so live features are
IDENTICAL to the data the model trained on. That is the fix for "everything
maps to one class": train and detect now share one extraction pipeline.

Requirements (in the same folder):
    landmarks.py        (from Gesto Labeller — extract_vector + normalize_vector)
    gesto_model.h5
    labels.json         (carries frames + feature_dim, written by train script)

Run:
    python gesto_labeller_detect.py --model gesto_model.h5 --labels labels.json

Press 'q' to quit.
"""

import json
import argparse
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Gesto Labeller's canonical extraction + normalization
from landmarks import extract_vector, normalize_vector, region_dim

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Map (region, hands-mode) by the feature dim stored in labels.json.
# 63 -> Hands/one, 126 -> Hands/two, 132 -> Pose, 32 -> Legs, 258 -> Full
DIM_TO_REGION = {
    63:  ("Hands", "one"),
    126: ("Hands", "two"),
    132: ("Pose", "two"),
    32:  ("Legs", "two"),
    258: ("Full", "two"),
}

# Optional display-name map, e.g. {"0": "A", "1": "B"}
DISPLAY_NAMES = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gesto_model.h5")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()

    model = load_model(args.model)
    meta = json.load(open(args.labels))
    index_to_name = {int(k): v for k, v in meta["labels"].items()}
    frames = int(meta["frames"])
    dim = int(meta["feature_dim"])

    if dim not in DIM_TO_REGION:
        raise RuntimeError(f"Unknown feature dim {dim}; can't pick a region.")
    region, hands = DIM_TO_REGION[dim]
    print(f"Model: {frames} frames x {dim} feats  ->  region '{region}' "
          f"(hands={hands})")

    def label_for(idx):
        raw = index_to_name[idx]
        return DISPLAY_NAMES.get(str(raw), raw)

    def label_for_name(raw):
        return DISPLAY_NAMES.get(str(raw), raw)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return

    seq = deque(maxlen=frames)
    recent = deque(maxlen=args.smooth)
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

            # draw whatever is relevant
            mp_drawing.draw_landmarks(frame, res.left_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, res.right_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)
            if region in ("Pose", "Legs", "Full"):
                mp_drawing.draw_landmarks(frame, res.pose_landmarks,
                                          mp_holistic.POSE_CONNECTIONS)

            # EXACT same extraction + normalization the Labeller used to save data
            raw = extract_vector(res, region, hands)
            if raw is None:
                feat = zero_vec
            else:
                feat = normalize_vector(raw, region, hands).astype(np.float32)
            seq.append(feat)

            if len(seq) == frames:
                probs = model.predict(np.expand_dims(np.array(seq), 0),
                                      verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])

                # per-frame printout: top + all class confidences
                ranked = sorted(
                    ((index_to_name[i], float(p)) for i, p in enumerate(probs)),
                    key=lambda kv: kv[1], reverse=True,
                )
                all_conf = "  ".join(f"{label_for_name(n)}:{p:.2f}"
                                     for n, p in ranked)
                print(f"[frame] top={label_for(idx)} ({conf:.2f})  |  {all_conf}")

                recent.append(idx if conf >= args.threshold else -1)
                if len(recent) == args.smooth:
                    top, count = Counter(recent).most_common(1)[0]
                    if top != -1 and count > args.smooth // 2:
                        shown_label, shown_conf = label_for(top), conf
                    else:
                        shown_label = ""

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (40, 24, 13), -1)
            text = f"{shown_label}  ({shown_conf:.2f})" if shown_label else "..."
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (39, 159, 239), 2, cv2.LINE_AA)

            cv2.imshow("Gesto - detect (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()