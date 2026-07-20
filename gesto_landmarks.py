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


# ---- drawing -------------------------------------------------------------

# 21-joint hand skeleton (MediaPipe hand connection pairs)
_HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

# full-body pose skeleton (subset of MediaPipe POSE_CONNECTIONS, the visible ones)
_POSE_BONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),      # arms + shoulders
    (11, 23), (12, 24), (23, 24),                          # torso
    (23, 25), (25, 27), (27, 29), (27, 31),                # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),                # right leg
]

# legs-only: indices are into the 8 saved leg points (hips, knees, ankles, feet)
# LEG_POSE_IDX order = [23,24,25,26,27,28,31,32] -> local 0..7
_LEG_BONES = [(0, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7), (0, 1)]

_ACCENT = (40, 160, 230)   # BGR — orange-ish dot
_BONE = (60, 180, 75)      # BGR — green bone


def _draw_points(frame, pts_xy, bones):
    import cv2
    h, w = frame.shape[:2]
    for a, b in bones:
        if a < len(pts_xy) and b < len(pts_xy):
            ax, ay = pts_xy[a]; bx, by = pts_xy[b]
            if (ax or ay) and (bx or by):       # skip zero (missing) points
                cv2.line(frame, (int(ax * w), int(ay * h)),
                         (int(bx * w), int(by * h)), _BONE, 2, cv2.LINE_AA)
    for (x, y) in pts_xy:
        if x or y:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, _ACCENT, -1, cv2.LINE_AA)


def _hand_xy(hand):
    return [(lm.x, lm.y) for lm in hand.landmark] if hand else []


def draw_region(frame, res, region, hands="two"):
    """Draw the skeleton for the model's region onto the BGR frame in place."""
    if region == "Hands":
        if hands == "one":
            single = res.right_hand_landmarks or res.left_hand_landmarks
            if single:
                _draw_points(frame, _hand_xy(single), _HAND_BONES)
        else:
            for h in (res.left_hand_landmarks, res.right_hand_landmarks):
                if h:
                    _draw_points(frame, _hand_xy(h), _HAND_BONES)

    elif region == "Pose":
        if res.pose_landmarks:
            pts = [(lm.x, lm.y) for lm in res.pose_landmarks.landmark]
            _draw_points(frame, pts, _POSE_BONES)

    elif region == "Legs":
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            pts = [(lms[i].x, lms[i].y) for i in LEG_POSE_IDX]   # 8 points
            _draw_points(frame, pts, _LEG_BONES)

    elif region == "Full":
        if res.pose_landmarks:
            pts = [(lm.x, lm.y) for lm in res.pose_landmarks.landmark]
            _draw_points(frame, pts, _POSE_BONES)
        for h in (res.left_hand_landmarks, res.right_hand_landmarks):
            if h:
                _draw_points(frame, _hand_xy(h), _HAND_BONES)