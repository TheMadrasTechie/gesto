"""
Gesto — detect (HandLandmarker, one hand, rolling window).

Live recognition using MediaPipe Tasks HandLandmarker — the SAME model your
Flutter plugin uses — so features match training exactly. Rolling window of the
last `frames` frames, with smoothing.

Requirements (same folder):
    hlm_common.py
    hand_landmarker.task
    gesto_model.h5
    labels.json

Run:
    python hlm_detect.py --model gesto_model.h5 --labels labels.json

Press 'q' to quit.
"""

import json
import time
import argparse
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from hlm_common import (make_landmarker, landmarks_to_vector,
                        normalize_vector, FEATURE_DIM)

DISPLAY_NAMES = {}


def load_meta(path):
    data = json.load(open(path))
    if isinstance(data, dict) and "labels" in data:
        labels = {int(k): v for k, v in data["labels"].items()}
        return labels, int(data.get("frames", 30))
    return {int(k): v for k, v in data.items()}, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gesto_model.h5")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()

    model = load_model(args.model)
    index_to_name, frames = load_meta(args.labels)

    def label_for(idx):
        raw = index_to_name[idx]
        return DISPLAY_NAMES.get(str(raw), raw)

    landmarker = make_landmarker(running_mode="video", num_hands=1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return

    seq = deque(maxlen=frames)
    recent = deque(maxlen=args.smooth)
    shown_label, shown_conf = "", 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)
        result = landmarker.detect_for_video(mp_image, ts)

        # draw landmarks if present
        if result.hand_landmarks:
            h, w = frame.shape[:2]
            for lm in result.hand_landmarks[0]:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3,
                           (117, 199, 250), -1)

        # extract + normalize exactly like training
        feat = normalize_vector(landmarks_to_vector(result))
        _buf = "FEAT_FULL: "
        for i in range(0, len(feat), 3):
            _buf += f"[{feat[i]:.3f},{feat[i+1]:.3f},{feat[i+2]:.3f}] "
        print(_buf)
        seq.append(feat)

        if len(seq) == frames:
            probs = model.predict(np.expand_dims(np.array(seq), 0),
                                  verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])

            ranked = sorted(((index_to_name[i], float(p))
                             for i, p in enumerate(probs)),
                            key=lambda kv: kv[1], reverse=True)
            allc = "  ".join(f"{DISPLAY_NAMES.get(str(n), n)}:{p:.2f}"
                             for n, p in ranked)
            print(f"[frame] top={label_for(idx)} ({conf:.2f})  |  {allc}")

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
