"""
Gesto — detect.

Loads the trained model and runs live gesture recognition from the webcam.
Uses a 30-frame sliding window (matching how the data was collected) and
shows the predicted class name + confidence on screen.

Needs:
    gesto_model.h5      (from train.py)
    labels.json         (index -> name)

Run:
    pip install tensorflow opencv-python mediapipe numpy
    python detect.py

Press 'q' to quit.
"""

import json

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

CAMERA_INDEX = 0
MODEL_PATH = "gesto_model.h5"
LABELS_PATH = "labels.json"
SEQUENCE_LENGTH = 30
FEATURE_DIM = 63
THRESHOLD = 0.7   # only show a prediction above this confidence

# Optional: map your collected numbers to display names here.
# e.g. {"0": "A", "1": "B"} — leave empty to show the raw labels.
DISPLAY_NAMES = {}


def extract_keypoints(results):
    """Right-hand landmarks as a flat (63,) vector, zeros if no hand."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32
        ).flatten()
    return np.zeros(FEATURE_DIM, dtype=np.float32)


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
        index_to_name = json.load(f)
    # json keys are strings; normalize to int index -> name
    index_to_name = {int(k): v for k, v in index_to_name.items()}

    def label_for(idx):
        raw = index_to_name[idx]
        return DISPLAY_NAMES.get(raw, raw)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    sequence = []
    current_label = ""
    current_conf = 0.0

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

            # maintain rolling window of the last 30 frames
            sequence.append(extract_keypoints(results))
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                probs = model.predict(
                    np.expand_dims(sequence, axis=0), verbose=0
                )[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                if conf >= THRESHOLD:
                    current_label = label_for(idx)
                    current_conf = conf
                else:
                    current_label = ""
                    current_conf = conf

            # header bar
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (40, 24, 13), -1)
            if current_label:
                text = f"{current_label}  ({current_conf:.2f})"
            else:
                text = "..."
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (39, 159, 239), 2, cv2.LINE_AA)

            cv2.imshow("Gesto — detect (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()