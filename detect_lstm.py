"""
detect_lstm.py — live motion (sequence) gesture detection.

Maintains a rolling window of the last T frames' landmark vectors and classifies
the MOTION with the LSTM model from train_lstm.py. T comes from labels.json, so
the window always matches how the model was trained.

Usage:
    python detect_lstm.py artifacts_lstm
    python detect_lstm.py artifacts_lstm --source path/to/video.mp4
    python detect_lstm.py artifacts_lstm --threshold 0.6

Press q to quit.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

from gesto_landmarks import extract_vector, make_holistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", help="Folder with lstm_model.keras + labels.json")
    ap.add_argument("--source", default="0", help="Webcam index or video path")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    art = Path(args.artifacts)
    meta = json.loads((art / "labels.json").read_text())
    labels = meta["labels"]; region = meta["region"]; hands = meta.get("hands", "two")
    T = meta["seq_len"]; D = meta["input_dim"]
    model = keras.models.load_model(art / "lstm_model.keras")
    print(f"Loaded LSTM model: {len(labels)} classes {labels}, "
          f"region={region}, hands={hands}, window T={T}, dim={D}")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    # rolling window of the last T frame-vectors; zero-filled until it fills up
    window = deque(maxlen=T)
    holistic = make_holistic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        vec = extract_vector(res, region, hands)
        window.append(vec if (vec is not None and vec.shape[0] == D)
                      else np.zeros(D, np.float32))

        label, prob = "—", 0.0
        if len(window) == T:
            seq = np.stack(window)[None, :, :]        # (1, T, D)
            probs = model.predict(seq, verbose=0)[0]
            i = int(np.argmax(probs)); prob = float(probs[i])
            if prob >= args.threshold:
                label = labels[i]

        fill = f"{len(window)}/{T}"
        cv2.putText(frame, f"{label}  {prob:.2f}  [{fill}]", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 160, 230), 2)
        cv2.imshow("Gesto — LSTM detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release(); cv2.destroyAllWindows(); holistic.close()


if __name__ == "__main__":
    main()
