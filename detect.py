"""
Gesto — detect (fixed frame count).

Loads the trained model and runs live recognition from the webcam using a
rolling window of the last NUM_FRAMES frames (matching how data was collected).
Shows the predicted class name + confidence on screen.

Needs:
    gesto_model.h5
    labels.json
NUM_FRAMES (in gesto_common.py) must match what you trained with.

Run:
    python detect.py

Press 'q' to quit.
"""

import json
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from gesto_common import extract_keypoints, NUM_FRAMES

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

CAMERA_INDEX = 0
MODEL_PATH = "gesto_model.h5"
LABELS_PATH = "labels.json"
THRESHOLD = 0.7          # only show predictions above this confidence
SMOOTH_WINDOW = 5        # require agreement over this many recent predictions

# Map collected numbers to display names, e.g. {"0": "A", "1": "B"}.
DISPLAY_NAMES = {"0": "A", "1": "B", "2": "C"}


def draw(frame, results):
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )


def main():
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        index_to_name = {int(k): v for k, v in json.load(f).items()}

    def label_for(idx):
        raw = index_to_name[idx]
        return DISPLAY_NAMES.get(raw, raw)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    sequence = deque(maxlen=NUM_FRAMES)
    recent = deque(maxlen=SMOOTH_WINDOW)
    shown_label = ""
    shown_conf = 0.0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            draw(frame, results)

            sequence.append(extract_keypoints(results))

            if len(sequence) == NUM_FRAMES:
                probs = model.predict(
                    np.expand_dims(np.array(sequence), axis=0), verbose=0
                )[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])

                # smoothing: only commit a label the recent window agrees on
                if conf >= THRESHOLD:
                    recent.append(idx)
                else:
                    recent.append(-1)

                if len(recent) == SMOOTH_WINDOW:
                    top, count = Counter(recent).most_common(1)[0]
                    if top != -1 and count > SMOOTH_WINDOW // 2:
                        shown_label = label_for(top)
                        shown_conf = conf
                    else:
                        shown_label = ""

            # header bar
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