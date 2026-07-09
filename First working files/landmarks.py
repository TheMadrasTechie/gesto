"""
Landmark extraction and normalisation from a MediaPipe Holistic result.

Feature dimension per region:
    Hands (two)  -> 2 x 21 x 3 = 126
    Hands (one)  -> 1 x 21 x 3 =  63
    Pose         -> 33 x 4      = 132   (x, y, z, visibility)
    Legs         ->  8 x 4      =  32
    Full         -> 132 + 126   = 258

For the Hands region, `hands` selects "one" or "two":
- "two": left then right, each 21x3 (missing hand = zeros).
- "one": whichever single hand is present (prefers right, falls back to left).
"""

from __future__ import annotations
import numpy as np

LEG_POSE_IDX = [23, 24, 25, 26, 27, 28, 31, 32]  # hips, knees, ankles, foot index

# Base region dims (Hands assumes two-hand; use region_dim() for the exact value).
REGION_DIM = {"Hands": 126, "Pose": 132, "Legs": len(LEG_POSE_IDX) * 4, "Full": 258}


def region_dim(region: str, hands: str = "two") -> int:
    """Exact feature dim, accounting for one- vs two-hand mode."""
    if region == "Hands":
        return 63 if hands == "one" else 126
    return REGION_DIM.get(region, 0)


def _pose_array(res, indices=None) -> np.ndarray:
    n = len(indices) if indices is not None else 33
    out = np.zeros((n, 4), dtype=np.float32)
    if res.pose_landmarks:
        lms = res.pose_landmarks.landmark
        idxs = indices if indices is not None else range(33)
        for j, i in enumerate(idxs):
            lm = lms[i]
            out[j] = (lm.x, lm.y, lm.z, lm.visibility)
    return out.reshape(-1)


def _hand_array(hand_landmarks) -> np.ndarray:
    out = np.zeros((21, 3), dtype=np.float32)
    if hand_landmarks:
        for i, lm in enumerate(hand_landmarks.landmark):
            out[i] = (lm.x, lm.y, lm.z)
    return out.reshape(-1)


def _single_hand(res):
    """The one hand to use in one-hand mode: prefer right, else left."""
    return res.right_hand_landmarks or res.left_hand_landmarks


def extract_vector(res, region: str, hands: str = "two") -> np.ndarray | None:
    """Feature vector for the chosen region, or None if nothing detected."""
    if region == "Hands":
        if hands == "one":
            h = _single_hand(res)
            if not h:
                return None
            return _hand_array(h)
        if not res.left_hand_landmarks and not res.right_hand_landmarks:
            return None
        return np.concatenate([_hand_array(res.left_hand_landmarks),
                               _hand_array(res.right_hand_landmarks)])
    if region == "Pose":
        if not res.pose_landmarks:
            return None
        return _pose_array(res)
    if region == "Legs":
        if not res.pose_landmarks:
            return None
        return _pose_array(res, LEG_POSE_IDX)
    if region == "Full":
        if not (res.pose_landmarks or res.left_hand_landmarks or res.right_hand_landmarks):
            return None
        return np.concatenate([_pose_array(res),
                               _hand_array(res.left_hand_landmarks),
                               _hand_array(res.right_hand_landmarks)])
    return None


def _normalize_one_hand(pts: np.ndarray) -> np.ndarray:
    """pts: (21,3) -> wrist-relative, scale-normalised."""
    if np.any(pts):
        pts = pts - pts[0]
        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 1e-6:
            pts = pts / scale
    return pts


def normalize_vector(vec: np.ndarray, region: str, hands: str = "two") -> np.ndarray:
    """Translation/scale-invariant. Visibility columns (pose) left untouched."""
    if region == "Hands":
        if hands == "one":
            return _normalize_one_hand(vec.reshape(21, 3).copy()).reshape(-1)
        pts = vec.reshape(2, 21, 3).copy()
        for h in range(2):
            pts[h] = _normalize_one_hand(pts[h])
        return pts.reshape(-1)
    if region in ("Pose", "Legs"):
        n = 33 if region == "Pose" else len(LEG_POSE_IDX)
        pts = vec.reshape(n, 4).copy()
        xyz = pts[:, :3]
        mask = np.any(xyz != 0, axis=1)
        if np.any(mask):
            xyz -= xyz[mask].mean(axis=0)
            scale = np.linalg.norm(xyz, axis=1).max()
            if scale > 1e-6:
                xyz /= scale
        pts[:, :3] = xyz
        return pts.reshape(-1)
    if region == "Full":
        return np.concatenate([normalize_vector(vec[:132], "Pose"),
                               normalize_vector(vec[132:], "Hands", "two")])
    return vec
