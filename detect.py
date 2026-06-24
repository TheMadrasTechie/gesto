"""
Gesto — detect (variable length).

Press-to-detect, mirroring how data is collected: press SPACE to start, perform
the gesture, press SPACE to stop. The captured frames (any length) are resampled
to TARGET_LEN and classified. The predicted name + confidence stays on screen.

Needs:
    gesto_model.h5
    labels.json

Run:
    python detect.py

Controls:
    SPACE  start / stop a gesture
    q      quit
"""

import json

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from gesto_common import extract_keypoints, resample_sequence, TARGET_LEN

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

CAMERA_INDEX = 0
MODEL_PATH = "gesto_model.h5"
LABELS_PATH = "labels.json"
THRESHOLD = 0.7
MIN_FRAMES = 2

# Map collected numbers to display names, e.g. {"0": "A", "1": "B"}.
DISPLAY_NAMES =  {"0": "Thumbs Up", "1": "all fingers shown", "2" : "Hand Rotate"} 


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

    recording = False
    buffer = []
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

            if recording:
                buffer.append(extract_keypoints(results))

            # header bar
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (40, 24, 13), -1)
            if recording:
                text = f"REC  frames {len(buffer)}   SPACE = stop"
                color = (60, 60, 255)
            elif current_label:
                text = f"{current_label}  ({current_conf:.2f})   SPACE = go"
                color = (39, 159, 239)
            else:
                text = "SPACE = perform a gesture    q = quit"
                color = (200, 200, 200)
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)

            cv2.imshow("Gesto - detect (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                if not recording:
                    buffer = []
                    recording = True
                    current_label = ""
                else:
                    recording = False
                    if len(buffer) >= MIN_FRAMES:
                        seq = resample_sequence(
                            np.array(buffer, dtype=np.float32), TARGET_LEN
                        )
                        probs = model.predict(
                            np.expand_dims(seq, axis=0), verbose=0
                        )[0]
                        idx = int(np.argmax(probs))
                        conf = float(probs[idx])
                        if conf >= THRESHOLD:
                            current_label = label_for(idx)
                        else:
                            current_label = "?"
                        current_conf = conf
                    else:
                        current_label = ""
                    buffer = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
