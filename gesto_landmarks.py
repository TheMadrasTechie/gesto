"""
gesto_landmarks.py — live landmark extraction for detection.

This mirrors EXACTLY how Gesto Labeller captures data, so a model trained on
Gesto data receives identically-shaped, identically-ordered vectors at
inference time. It uses MediaPipe Holistic (the same engine the app uses).

It's a self-contained copy so the training folder runs without importing the
app. If you prefer, you can instead import the app's own
gesto_studio.core.landmarks — the logic is the same.
"""

import numpy as np

LEG_POSE_IDX = [23, 24, 25, 26, 27, 28, 31, 32]


def _pose_array(res, indices=None):
    n = len(indices) if indices is not None else 33
    out = np.zeros((n, 4), np.float32)
    if res.pose_landmarks:
        lms = res.pose_landmarks.landmark
        idxs = indices if indices is not None else range(33)
        for j, i in enumerate(idxs):
            lm = lms[i]
            out[j] = (lm.x, lm.y, lm.z, lm.visibility)
    return out.reshape(-1)


def _hand_array(hand):
    out = np.zeros((21, 3), np.float32)
    if hand:
        for i, lm in enumerate(hand.landmark):
            out[i] = (lm.x, lm.y, lm.z)
    return out.reshape(-1)


def extract_vector(res, region, hands="two"):
    """Return the feature vector for `region`, or None if nothing was detected.

    Ordering/shape is identical to Gesto's capture so it matches training data.
    """
    if region == "Hands":
        if hands == "one":
            single = res.right_hand_landmarks or res.left_hand_landmarks
            if single is None:
                return None
            return _hand_array(single)                      # (63,)
        # two-hand: left then right, zeros for a missing hand
        if not (res.left_hand_landmarks or res.right_hand_landmarks):
            return None
        return np.concatenate([_hand_array(res.left_hand_landmarks),
                               _hand_array(res.right_hand_landmarks)])   # (126,)
    if region == "Pose":
        if not res.pose_landmarks:
            return None
        return _pose_array(res)                             # (132,)
    if region == "Legs":
        if not res.pose_landmarks:
            return None
        return _pose_array(res, LEG_POSE_IDX)               # (32,)
    if region == "Full":
        if not (res.pose_landmarks or res.left_hand_landmarks
                or res.right_hand_landmarks):
            return None
        return np.concatenate([                             # (258,)
            _pose_array(res),
            _hand_array(res.left_hand_landmarks),
            _hand_array(res.right_hand_landmarks)])
    return None


def make_holistic():
    """Create a MediaPipe Holistic engine (same settings as the app)."""
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)
