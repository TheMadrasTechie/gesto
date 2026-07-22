"""
Gesto — gesture explorer (Streamlit).

Pick a gesture class and every stored sample plays back at once as an animated
hand skeleton on a black screen, laid out in a grid. Pure viewer — no editing,
no webcam, no model.

Run:
    pip install streamlit numpy opencv-python pillow
    streamlit run viewer.py
"""

import os
import io

import numpy as np
import cv2
from PIL import Image
import streamlit as st

DATA_PATH = "gesture_data"
FEATURE_DIM = 63
TILE = 220               # pixel size of each sample tile
COLS = 4                 # grid columns
FRAME_MS = 70            # gif frame duration (ms) -> playback speed

# MediaPipe hand connections (pairs among the 21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

AMBER = (239, 159, 39)       # RGB
AMBER_DIM = (186, 117, 23)
POINT = (250, 199, 117)


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


def render_frame(features, size=TILE):
    """One (63,) frame -> RGB skeleton image on black."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    pts = np.array(features, dtype=np.float32).reshape(21, 3)

    if not np.any(pts):
        cv2.putText(img, "no hand", (size // 2 - 42, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        return img

    m = 24
    xs = (pts[:, 0] * (size - 2 * m) + m).astype(int)
    ys = (pts[:, 1] * (size - 2 * m) + m).astype(int)

    for a, b in HAND_CONNECTIONS:
        cv2.line(img, (xs[a], ys[a]), (xs[b], ys[b]), AMBER_DIM, 2, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(img, (xs[i], ys[i]), 3, POINT, -1, cv2.LINE_AA)
    cv2.circle(img, (xs[0], ys[0]), 5, AMBER, -1, cv2.LINE_AA)
    return img


@st.cache_data(show_spinner=False)
def build_gif(class_name, sample_file):
    """Render a sample's frames into a looping GIF (bytes).

    Each frame is drawn on an opaque black square and the GIF is saved without
    transparency, so tiles are fully black with no white show-through.
    """
    arr = np.load(os.path.join(DATA_PATH, class_name, sample_file))
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
        return None

    pil_frames = []
    for i in range(arr.shape[0]):
        rgb = render_frame(arr[i])               # already on black, (TILE,TILE,3)
        black = Image.new("RGB", (TILE, TILE), (0, 0, 0))
        black.paste(Image.fromarray(rgb), (0, 0))
        pil_frames.append(black)
    if not pil_frames:
        return None

    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True, append_images=pil_frames[1:],
        duration=FRAME_MS, loop=0, disposal=1,
    )
    return buf.getvalue()


def main():
    st.set_page_config(page_title="Gesto Explorer", layout="wide")
    # Force a black background everywhere and make every image tile a fixed
    # black square, so partially-sized GIFs never leave white gaps.
    st.markdown(
        """
        <style>
        .stApp { background-color: #000000; }
        div[data-testid="stImage"] {
            background-color: #000000;
            display: flex;
            justify-content: center;
            align-items: center;
            aspect-ratio: 1 / 1;
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stImage"] img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            background-color: #000000;
        }
        div[data-testid="stImage"] figcaption { color: #EF9F27 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='color:#EF9F27;margin-bottom:0;'>Gesto — Gesture Explorer</h2>",
        unsafe_allow_html=True,
    )

    classes = list_classes()
    if not classes:
        st.warning(f"No data in '{DATA_PATH}/'. Collect some gestures first.")
        return

    cls = st.selectbox("Select a gesture", classes)
    samples = list_samples(cls)
    st.caption(f"Class '{cls}'  ·  {len(samples)} samples (all looping)")

    if not samples:
        st.info("No samples in this class.")
        return

    cols = st.columns(COLS)
    for i, sample in enumerate(samples):
        gif = build_gif(cls, sample)
        with cols[i % COLS]:
            if gif is None:
                st.error(f"{sample}: bad shape")
            else:
                st.image(gif, caption=sample, use_column_width=True)


if __name__ == "__main__":
    main()