"""
Gesto — data collection (variable length).

Records gesture samples of their natural length. You press SPACE to start a
sample and SPACE again to stop, so a quick gesture might be 6 frames and a
slow one 40. Samples are stored at their raw length; resampling to a fixed
length happens at train/detect time.

You pass the class NUMBER on the command line.

Run:
    python collect_data.py 0
    python collect_data.py 1

Controls:
    SPACE  start / stop recording the current sample
    q      quit

Files:
    gesture_data/<class>/<sample_index>.npy     # shape (L, 63), L varies
"""

import os
import sys

import cv2
import numpy as np
import mediapipe as mp

from gesto_common import extract_keypoints, FEATURE_DIM

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

CAMERA_INDEX = 0
DATA_PATH = "gesture_data"
SAMPLES_PER_CLASS = 10
MIN_FRAMES = 2          # reject accidental too-short captures


def draw(frame, results):
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )


def banner(frame, text, y=40, color=(0, 165, 255)):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)


def main():
    if len(sys.argv) < 2:
        print("Usage: python collect_data.py <class_number>")
        return
    class_name = sys.argv[1]

    class_dir = os.path.join(DATA_PATH, class_name)
    os.makedirs(class_dir, exist_ok=True)

    existing = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
    sample_index = len(existing)
    if sample_index >= SAMPLES_PER_CLASS:
        print(f"Class '{class_name}' already has {sample_index} samples. Done.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    recording = False
    buffer = []

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened() and sample_index < SAMPLES_PER_CLASS:
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
                banner(frame, f"REC class '{class_name}'  sample "
                              f"{sample_index + 1}/{SAMPLES_PER_CLASS}  "
                              f"frames {len(buffer)}", color=(0, 0, 255))
                banner(frame, "SPACE = stop", y=70, color=(200, 200, 200))
            else:
                banner(frame, f"class '{class_name}'  next sample "
                              f"{sample_index + 1}/{SAMPLES_PER_CLASS}")
                banner(frame, "SPACE = start    q = quit", y=70,
                       color=(200, 200, 200))

            cv2.imshow("Gesto - collect (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                if not recording:
                    buffer = []
                    recording = True
                else:
                    recording = False
                    if len(buffer) >= MIN_FRAMES:
                        path = os.path.join(class_dir, f"{sample_index}.npy")
                        np.save(path, np.array(buffer, dtype=np.float32))
                        print(f"Saved {path}  length={len(buffer)} frames")
                        sample_index += 1
                    else:
                        print(f"Discarded sample (only {len(buffer)} frames).")
                    buffer = []

    cap.release()
    cv2.destroyAllWindows()
    print(f"Class '{class_name}': {sample_index}/{SAMPLES_PER_CLASS} samples.")


if __name__ == "__main__":
    main()
