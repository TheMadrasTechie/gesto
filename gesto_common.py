"""
Gesto — shared helpers.

Common logic used by collect / train / detect so they never drift apart:
  - extract_keypoints: one hand -> (63,) feature vector
  - resample_sequence: any (L, 63) gesture -> fixed (TARGET_LEN, 63)

Resampling is what makes variable-length gestures work: a gesture recorded
over 5, 12, or 40 frames is stretched/compressed to the same length, which
also makes the model speed-invariant (slow and fast versions look the same).
"""

import numpy as np

TARGET_LEN = 30      # every gesture is resampled to this many frames
FEATURE_DIM = 63     # 21 hand landmarks x (x, y, z)


def extract_keypoints(results):
    """Right-hand landmarks as a flat (63,) vector, zeros if no hand."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32
        ).flatten()
    return np.zeros(FEATURE_DIM, dtype=np.float32)


def resample_sequence(seq, target_len=TARGET_LEN):
    """Stretch/compress a (L, 63) gesture to (target_len, 63) by interpolation.

    L == target_len -> returned unchanged
    L == 1          -> the single frame is repeated (static gesture)
    otherwise       -> linear interpolation along the time axis per feature
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected (L, {FEATURE_DIM}), got {seq.shape}")

    L = seq.shape[0]
    if L == target_len:
        return seq
    if L == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    if L == 1:
        return np.repeat(seq, target_len, axis=0)

    old_idx = np.linspace(0.0, 1.0, L)
    new_idx = np.linspace(0.0, 1.0, target_len)
    out = np.stack(
        [np.interp(new_idx, old_idx, seq[:, j]) for j in range(FEATURE_DIM)],
        axis=1,
    )
    return out.astype(np.float32)
