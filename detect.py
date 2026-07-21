"""
detect.py — live LSTM detection matching your Alphabet notebook, any region.

Uses the exact rolling-window logic from your notebook:
    sequence.append(keypoints); sequence = sequence[-seq_len:]
    if len(sequence) == seq_len: predict
plus the 10-frame stability check before accepting a prediction. No padding —
this matches the fixed-length training above.

Usage:
    python detect.py artifacts_hands_right
    python detect.py artifacts_pose --source clip.mp4 --threshold 0.6

Press q to quit.
"""
import argparse, json
from collections import deque
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

from gesto_regions import REGIONS, draw_region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts")
    ap.add_argument("--source", default="0")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stable", type=int, default=10,
                    help="frames of agreement required before accepting (notebook uses 10)")
    args = ap.parse_args()

    art = Path(args.artifacts)
    meta = json.loads((art / "labels.json").read_text())
    labels = meta["labels"]; region_key = meta["region_key"]
    T = meta["seq_len"]; D = meta["input_dim"]
    extract = REGIONS[region_key]["extract"]
    model = load_model(art / "model.keras")
    print(f"Loaded {region_key} model: {labels} (T={T}, dim={D})")

    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")

    sequence = []            # rolling list, exactly like the notebook
    predictions = deque(maxlen=args.stable)
    current = "—"

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        draw_region(frame, results, region_key)

        keypoints = extract(results)
        sequence.append(keypoints)
        sequence = sequence[-T:]            # keep last T frames (notebook style)

        prob = 0.0
        if len(sequence) == T:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            top = int(np.argmax(res)); prob = float(res[top])
            predictions.append(top)
            # accept only if the last `stable` predictions all agree (notebook logic)
            if (len(predictions) == predictions.maxlen
                    and len(set(predictions)) == 1
                    and prob > args.threshold):
                current = labels[top]

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (245, 117, 16), -1)
        cv2.putText(frame, f"{current}   {prob:.2f}   [{len(sequence)}/{T}]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Gesto — LSTM detection", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release(); cv2.destroyAllWindows(); holistic.close()


if __name__ == "__main__":
    main()