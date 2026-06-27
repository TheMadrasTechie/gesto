"""
Gesto — landmark viewer (Streamlit).

Browses everything in gesture_data/ and plays each sample back as an animated
hand skeleton on a black screen. Pure landmark playback — no webcam, no model.

Run:
    pip install streamlit numpy opencv-python
    streamlit run viewer.py
"""

import os
import time

import cv2
import numpy as np
import streamlit as st

DATA_PATH = "gesture_data"
FEATURE_DIM = 63
CANVAS = 480                      # black canvas size (square)

# MediaPipe hand connections (pairs of the 21 landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]

AMBER = (39, 159, 239)   # BGR — Gesto amber
AMBER_DIM = (23, 117, 186)
POINT = (117, 199, 250)


def list_classes():
    if not os.path.isdir(DATA_PATH):
        return []
    return sorted(
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    )


def list_samples(class_name):
    d = os.path.join(DATA_PATH, class_name)
    return sorted(
        (f for f in os.listdir(d) if f.endswith(".npy")),
        key=lambda f: int(os.path.splitext(f)[0]) if f[:-4].isdigit() else 0,
    )


def draw_frame(features):
    """Render one (63,) frame as a skeleton on a black canvas (RGB)."""
    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    pts = np.array(features, dtype=np.float32).reshape(21, 3)

    # frame with no detection = all zeros
    if not np.any(pts):
        cv2.putText(img, "no hand", (CANVAS // 2 - 60, CANVAS // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # landmarks are normalized 0..1 (x, y); map to canvas with a margin
    m = 40
    xs = (pts[:, 0] * (CANVAS - 2 * m) + m).astype(int)
    ys = (pts[:, 1] * (CANVAS - 2 * m) + m).astype(int)

    for a, b in HAND_CONNECTIONS:
        cv2.line(img, (xs[a], ys[a]), (xs[b], ys[b]), AMBER_DIM, 2, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(img, (xs[i], ys[i]), 4, POINT, -1, cv2.LINE_AA)
    # wrist a touch bigger
    cv2.circle(img, (xs[0], ys[0]), 6, AMBER, -1, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    st.set_page_config(page_title="Gesto Viewer", layout="centered")
    st.markdown(
        "<h2 style='color:#EF9F27;'>Gesto — Landmark Viewer</h2>",
        unsafe_allow_html=True,
    )

    classes = list_classes()
    if not classes:
        st.warning(f"No data found in '{DATA_PATH}/'. Collect some gestures first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        cls = st.selectbox("Class", classes)
    samples = list_samples(cls)
    with col2:
        if not samples:
            st.info("No samples in this class.")
            return
        sample = st.selectbox("Sample", samples)

    arr = np.load(os.path.join(DATA_PATH, cls, sample))
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
        st.error(f"Unexpected shape {arr.shape}; expected (frames, 63).")
        return

    n_frames = arr.shape[0]
    st.caption(f"Class '{cls}'  ·  {sample}  ·  {n_frames} frames")

    fps = st.slider("Playback speed (fps)", 2, 30, 12)
    play = st.button("▶ Play")

    placeholder = st.empty()

    if play:
        for i in range(n_frames):
            placeholder.image(draw_frame(arr[i]),
                              caption=f"frame {i + 1}/{n_frames}")
            time.sleep(1.0 / fps)
    else:
        # show first frame by default, plus a manual scrubber
        idx = st.slider("Frame", 0, n_frames - 1, 0)
        placeholder.image(draw_frame(arr[idx]),
                          caption=f"frame {idx + 1}/{n_frames}")


if __name__ == "__main__":
    main()