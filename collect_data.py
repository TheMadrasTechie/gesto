"""
Gesto — data collection.

Collects gesture samples for one numeric class at a time.
  - 30 frames per sample
  - 10 samples per class
  - each sample saved as one array of shape (30, 63)
    (21 hand landmarks x [x, y, z])

You pass the class NUMBER on the command line; names get mapped in later
at display time.

Run:
    python collect_data.py 0          # collect class "0"
    python collect_data.py 1          # collect class "1"
    ...

Controls during capture:
    SPACE  start recording the next sample (30 frames)
    q      quit early

Files land in:
    gesture_data/<class>/<sample_index>.npy
"""

import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

CAMERA_INDEX = 0
DATA_PATH = "gesture_data"
FRAMES_PER_SAMPLE = 30
SAMPLES_PER_CLASS = 10
FEATURE_DIM = 21 * 3  # 63


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

    # resume from whatever samples already exist
    existing = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
    start_index = len(existing)
    if start_index >= SAMPLES_PER_CLASS:
        print(f"Class '{class_name}' already has {start_index} samples. Done.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    sample_index = start_index
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
                              f"frame {len(buffer)}/{FRAMES_PER_SAMPLE}",
                        color=(0, 0, 255))
                if len(buffer) == FRAMES_PER_SAMPLE:
                    path = os.path.join(class_dir, f"{sample_index}.npy")
                    np.save(path, np.array(buffer, dtype=np.float32))
                    print(f"Saved {path}  shape={np.array(buffer).shape}")
                    sample_index += 1
                    buffer = []
                    recording = False
                    time.sleep(0.3)  # brief pause between samples
            else:
                banner(frame, f"class '{class_name}'  next sample "
                              f"{sample_index + 1}/{SAMPLES_PER_CLASS}")
                banner(frame, "SPACE = record    q = quit", y=70,
                       color=(200, 200, 200))

            cv2.imshow("Gesto — collect (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and not recording:
                buffer = []
                recording = True

    cap.release()
    cv2.destroyAllWindows()
    print(f"Class '{class_name}': {sample_index}/{SAMPLES_PER_CLASS} samples collected.")


if __name__ == "__main__":
    main()