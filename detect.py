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

from gesto_regions import REGIONS, draw_region, normalize_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts")
    ap.add_argument("--source", default="0")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stable", type=int, default=10,
                    help="frames of agreement required before accepting (notebook uses 10)")
    ap.add_argument("--raw", action="store_true",
                    help="Use if your Gesto data was captured with 'Normalise' "
                         "UNCHECKED. Default assumes normalized (Gesto's default).")
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
    # Gesto Labeller MIRRORS the webcam when capturing (but not video files).
    # We must mirror here too, or the model sees horizontally-flipped hands at
    # detection vs training -> wrong predictions. Match Gesto exactly.
    is_webcam = isinstance(src, int)

    sequence = []            # rolling list, exactly like the notebook
    predictions = deque(maxlen=args.stable)
    current = "—"            # committed label (after stability check)
    # a distinct colour per class for the probability bars
    rng = np.random.RandomState(7)
    colors = [(int(rng.randint(60, 256)), int(rng.randint(60, 256)),
               int(rng.randint(60, 256))) for _ in labels]

    def draw_prob_bars(frame, probs):
        """Draw one horizontal bar per class with its live percentage."""
        for i, p in enumerate(probs):
            y = 60 + i * 34
            # bar background
            cv2.rectangle(frame, (10, y), (10 + 260, y + 26), (50, 50, 50), -1)
            # bar fill proportional to probability
            cv2.rectangle(frame, (10, y), (10 + int(260 * float(p)), y + 26),
                          colors[i], -1)
            cv2.putText(frame, f"{labels[i]}  {float(p) * 100:4.1f}%",
                        (16, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

    probs = np.zeros(len(labels), np.float32)   # last prediction, shown live

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if is_webcam:
            frame = cv2.flip(frame, 1)      # mirror, matching Gesto capture
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        draw_region(frame, results, region_key)

        keypoints = extract(results)
        # match Gesto's capture: it normalizes by default, so normalize here too
        if not args.raw:
            keypoints = normalize_vector(keypoints, region_key)
        sequence.append(keypoints)
        sequence = sequence[-T:]            # keep last T frames (notebook style)

        # keep predicting every frame the window is full, so the bars are live
        if len(sequence) == T:
            probs = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            top = int(np.argmax(probs))
            predictions.append(top)
            # the "committed" label still uses the stability check (steady output)
            if (len(predictions) == predictions.maxlen
                    and len(set(predictions)) == 1
                    and float(probs[top]) > args.threshold):
                current = labels[top]

        # header: the committed label + window fill
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (245, 117, 16), -1)
        cv2.putText(frame, f"{current}    [{len(sequence)}/{T}]",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        # live probability bars for ALL classes
        draw_prob_bars(frame, probs)

        cv2.imshow("Gesto — LSTM detection", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release(); cv2.destroyAllWindows(); holistic.close()


if __name__ == "__main__":
    main()