"""
gesto_regions.py — region definitions shared by the per-region train/detect
scripts. Matches Gesto Labeller's capture exactly (MediaPipe Holistic).

Each region has: a feature dimension, a keypoint extractor, and a drawer.
This mirrors your Alphabet notebook's extract_keypoints(), generalized to all
five region types you asked for:

    hands_right : right hand only              -> 63   (21 x 3)   [notebook style]
    hands_two   : left + right hand            -> 126  (2 x 21 x 3)
    pose        : full body pose               -> 132  (33 x 4: x,y,z,visibility)
    legs        : lower body subset            -> 32   (8 x 4)
    full        : pose + both hands            -> 258  (132 + 126)
"""

import numpy as np

LEG_POSE_IDX = [23, 24, 25, 26, 27, 28, 31, 32]

# ---- keypoint extractors (return a flat float32 vector) ----

def _rh(results):
    return (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3, np.float32))

def _lh(results):
    return (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3, np.float32))

def _pose_full(results):
    return (np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4, np.float32))

def _pose_legs(results):
    if not results.pose_landmarks:
        return np.zeros(len(LEG_POSE_IDX) * 4, np.float32)
    lm = results.pose_landmarks.landmark
    return np.array([[lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]
                     for i in LEG_POSE_IDX]).flatten()

def extract_hands_right(results): return _rh(results).astype(np.float32)
def extract_hands_two(results):   return np.concatenate([_lh(results), _rh(results)]).astype(np.float32)
def extract_pose(results):        return _pose_full(results).astype(np.float32)
def extract_legs(results):        return _pose_legs(results).astype(np.float32)
def extract_full(results):
    return np.concatenate([_pose_full(results), _lh(results), _rh(results)]).astype(np.float32)


REGIONS = {
    "hands_right": {"dim": 63,  "extract": extract_hands_right, "gesto_region": "Hands", "hands": "one"},
    "hands_two":   {"dim": 126, "extract": extract_hands_two,   "gesto_region": "Hands", "hands": "two"},
    "pose":        {"dim": 132, "extract": extract_pose,         "gesto_region": "Pose",  "hands": "two"},
    "legs":        {"dim": 32,  "extract": extract_legs,         "gesto_region": "Legs",  "hands": "two"},
    "full":        {"dim": 258, "extract": extract_full,         "gesto_region": "Full",  "hands": "two"},
}


# ---- drawing (BGR frame, in place) ----

_HAND_BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
               (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),
               (18,19),(19,20),(0,17)]
_POSE_BONES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
               (23,25),(25,27),(27,29),(27,31),(24,26),(26,28),(28,30),(28,32)]
_LEG_BONES  = [(0,2),(2,4),(4,6),(1,3),(3,5),(5,7),(0,1)]


def _dots(frame, pts, bones):
    import cv2
    h, w = frame.shape[:2]
    for a, b in bones:
        if a < len(pts) and b < len(pts):
            ax, ay = pts[a]; bx, by = pts[b]
            if (ax or ay) and (bx or by):
                cv2.line(frame, (int(ax*w), int(ay*h)), (int(bx*w), int(by*h)), (60,180,75), 2, cv2.LINE_AA)
    for x, y in pts:
        if x or y:
            cv2.circle(frame, (int(x*w), int(y*h)), 3, (40,160,230), -1, cv2.LINE_AA)


def draw_region(frame, results, region_key):
    def hand_xy(h): return [(lm.x, lm.y) for lm in h.landmark] if h else []
    if region_key in ("hands_right", "hands_two"):
        if results.right_hand_landmarks:
            _dots(frame, hand_xy(results.right_hand_landmarks), _HAND_BONES)
        if region_key == "hands_two" and results.left_hand_landmarks:
            _dots(frame, hand_xy(results.left_hand_landmarks), _HAND_BONES)
    elif region_key == "pose" and results.pose_landmarks:
        _dots(frame, [(lm.x, lm.y) for lm in results.pose_landmarks.landmark], _POSE_BONES)
    elif region_key == "legs" and results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        _dots(frame, [(lm[i].x, lm[i].y) for i in LEG_POSE_IDX], _LEG_BONES)
    elif region_key == "full":
        if results.pose_landmarks:
            _dots(frame, [(lm.x, lm.y) for lm in results.pose_landmarks.landmark], _POSE_BONES)
        for h in (results.left_hand_landmarks, results.right_hand_landmarks):
            if h: _dots(frame, hand_xy(h), _HAND_BONES)
