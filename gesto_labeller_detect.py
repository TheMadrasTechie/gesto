"""
Gesto — detect for Gesto Labeller models (Holistic, any region).

Runs live recognition using a model trained by gesto_labeller_train.py. It reads
the feature dim + frame count from labels.json and extracts the MATCHING region
from MediaPipe Holistic, so it works for Hands(126)/Pose(132)/Legs(32)/Full(258).

Needs:
    gesto_model.h5
    labels.json     (written by gesto_labeller_train.py; carries frames + dim)

Run:
    python gesto_labeller_detect.py
    python gesto_labeller_detect.py --model asl_model.h5 --labels labels.json

Press 'q' to quit.
"""

import json
import argparse
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Landmark counts per region of MediaPipe Holistic
POSE_N = 33      # x,y,z,visibility -> 4 each
HAND_N = 21      # x,y,z            -> 3 each

# Which feature dim corresponds to which region (matches Gesto Labeller)
DIM_TO_REGION = {63: "hand", 126: "hands", 132: "pose", 32: "legs", 258: "full"}

# Optional display-name map, e.g. {"0": "A", "1": "B"}
DISPLAY_NAMES = {}


def _pose_xyzv(res):
    if res.pose_landmarks:
        return np.array([[l.x, l.y, l.z, l.visibility]
                         for l in res.pose_landmarks.landmark], np.float32).flatten()
    return np.zeros(POSE_N * 4, np.float32)


def _hand_xyz(landmarks):
    if landmarks:
        return np.array([[l.x, l.y, l.z] for l in landmarks.landmark],
                        np.float32).flatten()
    return np.zeros(HAND_N * 3, np.float32)


def extract_region(res, region):
    """Return the feature vector for the requested region from Holistic results."""
    if region == "hand":
        # single hand, 63 features: prefer right hand, fall back to left
        if res.right_hand_landmarks:
            return _hand_xyz(res.right_hand_landmarks)
        if res.left_hand_landmarks:
            return _hand_xyz(res.left_hand_landmarks)
        return np.zeros(HAND_N * 3, np.float32)               # 63
    if region == "hands":
        lh = _hand_xyz(res.left_hand_landmarks)
        rh = _hand_xyz(res.right_hand_landmarks)
        return np.concatenate([lh, rh])                       # 126
    if region == "pose":
        return _pose_xyzv(res)                                # 132
    if region == "legs":
        # legs = 8 lower-body pose joints (indices 23..30) x (x,y,z,vis) = 32
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            return np.array([[lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]
                             for i in range(23, 31)], np.float32).flatten()
        return np.zeros(32, np.float32)
    if region == "full":
        return np.concatenate([_pose_xyzv(res),
                               _hand_xyz(res.left_hand_landmarks),
                               _hand_xyz(res.right_hand_landmarks)])   # 258
    raise ValueError(f"unknown region {region}")


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
    region = DIM_TO_REGION.get(dim)
    if region is None:
        raise RuntimeError(f"Unknown feature dim {dim}; can't pick a region.")
    print(f"Model: {frames} frames x {dim} feats  ->  region '{region}'")

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
            if region in ("pose", "legs", "full"):
                mp_drawing.draw_landmarks(frame, res.pose_landmarks,
                                          mp_holistic.POSE_CONNECTIONS)

            seq.append(extract_region(res, region))

            if len(seq) == frames:
                probs = model.predict(np.expand_dims(np.array(seq), 0),
                                      verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])

                # print this frame's top prediction + every class confidence
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