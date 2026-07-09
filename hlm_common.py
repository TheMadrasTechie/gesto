"""
Gesto — shared HandLandmarker (Tasks API) extraction + normalization.

Uses MediaPipe Tasks HandLandmarker (the .task model), so features match a
Flutter app using the hand_landmarker plugin. Train and detect BOTH import this
so their features are guaranteed identical.

One hand, 63 features (21 landmarks x, y, z).

You need the model file 'hand_landmarker.task' in the same folder. Download:
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FEATURE_DIM = 63
MODEL_TASK = "hand_landmarker.task"


def make_landmarker(running_mode="image", num_hands=1):
    """Create a HandLandmarker. running_mode: 'image' (per-frame) or 'video'."""
    base = mp_python.BaseOptions(model_asset_path=MODEL_TASK)
    mode = (vision.RunningMode.VIDEO if running_mode == "video"
            else vision.RunningMode.IMAGE)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=mode,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def landmarks_to_vector(result):
    """First detected hand -> flat (63,) x,y,z vector; zeros if none."""
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand],
                        dtype=np.float32).flatten()
    return np.zeros(FEATURE_DIM, dtype=np.float32)


def normalize_vector(vec):
    """Wrist-relative, scale-normalized (matches the Labeller convention).

    Subtract wrist (landmark 0), divide by the max distance from wrist to any
    landmark. IMPORTANT: replicate this exactly in Dart for the Flutter app.
    """
    pts = np.asarray(vec, dtype=np.float32).reshape(21, 3)
    if np.any(pts):
        pts = pts - pts[0]
        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 1e-6:
            pts = pts / scale
    return pts.flatten().astype(np.float32)


def normalize_sequence(seq):
    seq = np.asarray(seq, dtype=np.float32)
    return np.stack([normalize_vector(seq[i]) for i in range(seq.shape[0])])
