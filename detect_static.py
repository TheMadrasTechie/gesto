"""
detect_static.py — live single-frame gesture detection.

Runs your webcam (or a video), extracts the same landmark vector Gesto captured,
and classifies EACH FRAME with the static model from train_static.py.

Usage:
    python detect_static.py artifacts_static
    python detect_static.py artifacts_static --source path/to/video.mp4
    python detect_static.py artifacts_static --threshold 0.6

'artifacts_static' is the folder containing static_model.keras + labels.json.
Press q to quit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

from gesto_landmarks import extract_vector, make_holistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", help="Folder with static_model.keras + labels.json")
    ap.add_argument("--source", default="0", help="Webcam index or video path")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Min softmax prob to show a prediction")
    args = ap.parse_args()

    art = Path(args.artifacts)
    meta = json.loads((art / "labels.json").read_text())
    labels = meta["labels"]; region = meta["region"]; hands = meta.get("hands", "two")
    model = keras.models.load_model(art / "static_model.keras")
    print(f"Loaded static model: {len(labels)} classes {labels}, "
          f"region={region}, hands={hands}")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    holistic = make_holistic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        vec = extract_vector(res, region, hands)

        label, prob = "—", 0.0
        if vec is not None and vec.shape[0] == meta["input_dim"]:
            probs = model.predict(vec[None, :], verbose=0)[0]
            i = int(np.argmax(probs)); prob = float(probs[i])
            if prob >= args.threshold:
                label = labels[i]

        cv2.putText(frame, f"{label}  {prob:.2f}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 160, 230), 2)
        cv2.imshow("Gesto — static detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release(); cv2.destroyAllWindows(); holistic.close()


if __name__ == "__main__":
    main()
