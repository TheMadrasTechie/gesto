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

from gesto_landmarks import extract_vector, make_holistic, draw_region


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

    # Rolling buffer of the most recent real frame-vectors. We keep up to T of
    # them and build the model input padded the SAME way training pads:
    # real frames first, zeros after (post-padding). This is the key fix —
    # a deque alone feeds a different temporal layout than the model trained on.
    buffer = deque(maxlen=T)
    # smooth the output over a few frames so the label doesn't flicker
    recent = deque(maxlen=5)
    holistic = make_holistic()
    MIN_FRAMES = max(3, T // 4)   # need a few real frames before predicting

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        vec = extract_vector(res, region, hands)

        # draw the skeleton for this project's region onto the frame
        draw_region(frame, res, region, hands)

        # only append frames where something was actually detected; a gap in
        # detection shouldn't inject zero-rows in the middle of a motion
        if vec is not None and vec.shape[0] == D:
            buffer.append(vec)

        label, prob = "—", 0.0
        if len(buffer) >= MIN_FRAMES:
            # build a (T, D) input: real frames first, zero-pad the rest —
            # identical to how load_sequence() padded during training
            seq = np.zeros((T, D), np.float32)
            frames = list(buffer)[-T:]
            seq[:len(frames)] = frames
            probs = model.predict(seq[None, :, :], verbose=0)[0]
            i = int(np.argmax(probs)); p = float(probs[i])
            recent.append((i, p))
            # majority vote over the recent window, averaged confidence
            from collections import Counter
            votes = Counter(idx for idx, _ in recent)
            best_i, _ = votes.most_common(1)[0]
            avg_p = float(np.mean([pp for idx, pp in recent if idx == best_i]))
            if avg_p >= args.threshold:
                label, prob = labels[best_i], avg_p

        fill = f"{len(buffer)}/{T}"
        cv2.putText(frame, f"{label}  {prob:.2f}  [{fill}]", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 160, 230), 2)
        cv2.putText(frame, "r=reset  q=quit", (12, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Gesto — LSTM detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):          # clear the buffer to start a fresh gesture
            buffer.clear(); recent.clear()

    cap.release(); cv2.destroyAllWindows(); holistic.close()


if __name__ == "__main__":
    main()