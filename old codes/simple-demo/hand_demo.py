"""
Gesto — hand landmark demo.

Opens the webcam (camera index 1), runs MediaPipe Hands, and draws the
21 landmarks + connections on each detected hand in real time.

Run:
    pip install opencv-python mediapipe
    python hand_demo.py

Press 'q' to quit.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

CAMERA_INDEX = 0


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Empty frame, skipping.")
                continue

            # Mirror for a natural selfie view
            frame = cv2.flip(frame, 1)

            # MediaPipe wants RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            # Draw on the original BGR frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            cv2.imshow("Gesto — hand demo (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()