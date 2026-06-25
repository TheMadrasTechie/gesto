"""
Gesto — shared helpers.

Common logic used by collect / train / detect so they never drift apart.

NUM_FRAMES is the single knob: set it to 10, 20, or 30 and every script uses
that fixed sequence length. Change it here and recollect + retrain.
"""

import numpy as np

# ---- the one setting that controls everything ----
NUM_FRAMES = 30        # frames per gesture sample (try 10, 20, or 30)
# --------------------------------------------------

FEATURE_DIM = 63       # 21 hand landmarks x (x, y, z)


def extract_keypoints(results):
    """Right-hand landmarks as a flat (63,) vector, zeros if no hand."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32
        ).flatten()
    return np.zeros(FEATURE_DIM, dtype=np.float32)
