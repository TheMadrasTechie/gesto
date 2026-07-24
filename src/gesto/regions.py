"""
Region definitions — extraction, normalization and drawing for each region.

These mirror Gesto Labeller's capture exactly, so a model trained on Gesto data
receives identically-shaped, identically-ordered, identically-normalized vectors
at inference time.

    hands_one : one hand (prefer right, else left)  ->  63   (21 x 3)
    hands_two : left hand + right hand              -> 126   (2 x 21 x 3)
    pose      : full body                           -> 132   (33 x 4: x,y,z,vis)
    legs      : lower body subset                   ->  32   (8 x 4)
    full      : pose + both hands                   -> 258   (132 + 126)
"""

from __future__ import annotations

import numpy as np

LEG_POSE_IDX = [23, 24, 25, 26, 27, 28, 31, 32]

REGION_KEYS = ("hands_one", "hands_two", "pose", "legs", "full")

# region -> (feature dim, Gesto project "region", Gesto project "hands")
REGION_INFO = {
    "hands_one": (63, "Hands", "one"),
    "hands_two": (126, "Hands", "two"),
    "pose": (132, "Pose", "two"),
    "legs": (32, "Legs", "two"),
    "full": (258, "Full", "two"),
}


def feature_dim(region: str) -> int:
    _check(region)
    return REGION_INFO[region][0]


def _check(region: str) -> None:
    if region not in REGION_INFO:
        raise ValueError(
            f"Unknown region {region!r}. Choose one of: {', '.join(REGION_KEYS)}"
        )


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def _hand(landmarks) -> np.ndarray:
    if landmarks is None:
        return np.zeros(63, np.float32)
    return np.array([[p.x, p.y, p.z] for p in landmarks.landmark],
                    dtype=np.float32).reshape(-1)


def _pose_all(res) -> np.ndarray:
    if not res.pose_landmarks:
        return np.zeros(132, np.float32)
    return np.array([[p.x, p.y, p.z, p.visibility]
                     for p in res.pose_landmarks.landmark],
                    dtype=np.float32).reshape(-1)


def _pose_legs(res) -> np.ndarray:
    if not res.pose_landmarks:
        return np.zeros(len(LEG_POSE_IDX) * 4, np.float32)
    lm = res.pose_landmarks.landmark
    return np.array([[lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]
                     for i in LEG_POSE_IDX], dtype=np.float32).reshape(-1)


def extract(res, region: str) -> np.ndarray | None:
    """Feature vector for `region`, or None when nothing was detected.

    Returning None (rather than zeros) lets callers skip empty frames instead of
    feeding the model meaningless all-zero input.
    """
    _check(region)
    if region == "hands_one":
        hand = res.right_hand_landmarks or res.left_hand_landmarks
        if hand is None:
            return None
        return _hand(hand)
    if region == "hands_two":
        if not (res.left_hand_landmarks or res.right_hand_landmarks):
            return None
        return np.concatenate([_hand(res.left_hand_landmarks),
                               _hand(res.right_hand_landmarks)])
    if region == "pose":
        return _pose_all(res) if res.pose_landmarks else None
    if region == "legs":
        return _pose_legs(res) if res.pose_landmarks else None
    # full
    if not (res.pose_landmarks or res.left_hand_landmarks
            or res.right_hand_landmarks):
        return None
    return np.concatenate([_pose_all(res),
                           _hand(res.left_hand_landmarks),
                           _hand(res.right_hand_landmarks)])


# --------------------------------------------------------------------------
# normalization — byte-for-byte identical to Gesto Labeller's normalize_vector
# --------------------------------------------------------------------------

def _norm_hand(pts: np.ndarray) -> np.ndarray:
    """(21,3) -> wrist-relative, scale-normalised."""
    if np.any(pts):
        pts = pts - pts[0]
        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 1e-6:
            pts = pts / scale
    return pts


def _norm_body(pts: np.ndarray) -> np.ndarray:
    """(N,4) x,y,z,visibility -> centred and scale-normalised; visibility kept."""
    xyz = pts[:, :3]
    mask = np.any(xyz != 0, axis=1)
    if np.any(mask):
        xyz -= xyz[mask].mean(axis=0)
        scale = np.linalg.norm(xyz, axis=1).max()
        if scale > 1e-6:
            xyz /= scale
    pts[:, :3] = xyz
    return pts


def normalize(vec, region: str) -> np.ndarray:
    """Translation/scale-invariant normalization, matching Gesto exactly."""
    _check(region)
    vec = np.asarray(vec, np.float32)
    if region == "hands_one":
        return _norm_hand(vec.reshape(21, 3).copy()).reshape(-1)
    if region == "hands_two":
        pts = vec.reshape(2, 21, 3).copy()
        for h in range(2):
            pts[h] = _norm_hand(pts[h])
        return pts.reshape(-1)
    if region in ("pose", "legs"):
        n = 33 if region == "pose" else len(LEG_POSE_IDX)
        return _norm_body(vec.reshape(n, 4).copy()).reshape(-1)
    # full: pose part as body, hands part per-hand
    pose = _norm_body(vec[:132].reshape(33, 4).copy()).reshape(-1)
    hands = vec[132:].reshape(2, 21, 3).copy()
    for h in range(2):
        hands[h] = _norm_hand(hands[h])
    return np.concatenate([pose, hands.reshape(-1)])


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
POSE_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
              (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
              (24, 26), (26, 28), (28, 30), (28, 32)]
LEG_BONES = [(0, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7), (0, 1)]

_BONE_COLOR = (60, 180, 75)
_JOINT_COLOR = (40, 160, 230)


def _plot(frame, pts, bones) -> None:
    import cv2
    h, w = frame.shape[:2]
    for a, b in bones:
        if a < len(pts) and b < len(pts):
            ax, ay = pts[a]
            bx, by = pts[b]
            if (ax or ay) and (bx or by):
                cv2.line(frame, (int(ax * w), int(ay * h)),
                         (int(bx * w), int(by * h)), _BONE_COLOR, 2, cv2.LINE_AA)
    for x, y in pts:
        if x or y:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, _JOINT_COLOR, -1,
                       cv2.LINE_AA)


def draw(frame, res, region: str) -> None:
    """Draw the region's skeleton onto a BGR frame, in place."""
    _check(region)
    xy = lambda h: [(p.x, p.y) for p in h.landmark] if h else []

    if region == "hands_one":
        hand = res.right_hand_landmarks or res.left_hand_landmarks
        if hand:
            _plot(frame, xy(hand), HAND_BONES)
    elif region == "hands_two":
        for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
            if hand:
                _plot(frame, xy(hand), HAND_BONES)
    elif region == "pose":
        if res.pose_landmarks:
            _plot(frame, xy(res.pose_landmarks), POSE_BONES)
    elif region == "legs":
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            _plot(frame, [(lm[i].x, lm[i].y) for i in LEG_POSE_IDX], LEG_BONES)
    else:  # full
        if res.pose_landmarks:
            _plot(frame, xy(res.pose_landmarks), POSE_BONES)
        for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
            if hand:
                _plot(frame, xy(hand), HAND_BONES)


def make_holistic():
    """MediaPipe Holistic with the same settings Gesto Labeller captures with."""
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)
