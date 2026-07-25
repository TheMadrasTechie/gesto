"""
Live detection from a webcam or video file.

Matches Gesto Labeller's capture exactly, which is what keeps predictions
correct: the same MediaPipe Holistic engine, the same landmark ordering, the
same normalization, and the same webcam mirroring.

static   -> classifies every frame on its own; instant, no warm-up.
sequence -> keeps a rolling window of the last seq_len frames.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from . import artifacts
from .regions import draw, extract, make_holistic

HEADER_BG = (245, 117, 16)
WHITE = (255, 255, 255)


class Predictor:
    """A trained model plus everything needed to feed it correctly."""

    def __init__(self, run: str | Path):
        from ._compat import keras
        k = keras()
        self.run = Path(run)
        self.meta = artifacts.load_meta(self.run)
        self.model = k.models.load_model(artifacts.model_path(self.run))
        self.labels: list[str] = self.meta["labels"]
        self.region: str = self.meta["region"]
        self.mode: str = self.meta["mode"]
        self.dim: int = self.meta["input_dim"]
        self.normalized: bool = self.meta.get("normalized", True)
        self.seq_len: int = int(self.meta.get("seq_len", 1))
        self._window: deque = deque(maxlen=self.seq_len)

    @classmethod
    def load(cls, mode: str, region: str, *,
             root: str | Path = artifacts.DEFAULT_ROOT,
             version: int | str | None = None) -> "Predictor":
        return cls(artifacts.resolve(root, mode, region, version))

    def features(self, res) -> np.ndarray | None:
        """Landmark vector for a frame, normalized the way training expected."""
        vec = extract(res, self.region)
        if vec is None:
            return None
        if self.normalized:
            from .regions import normalize
            vec = normalize(vec, self.region)
        return vec

    def reset(self) -> None:
        self._window.clear()

    def predict(self, vec: np.ndarray | None) -> np.ndarray | None:
        """Probabilities for one frame, or None when there's nothing to say yet.

        For sequence models this returns None until the window is full.
        """
        if self.mode == "static":
            if vec is None:
                return None
            return self.model.predict(vec[None, :], verbose=0)[0]

        if vec is not None:
            self._window.append(vec)
        if len(self._window) < self.seq_len:
            return None
        batch = np.stack(self._window)[None, :, :]
        return self.model.predict(batch, verbose=0)[0]

    @property
    def progress(self) -> tuple[int, int]:
        return len(self._window), self.seq_len


def _palette(n: int) -> list[tuple[int, int, int]]:
    rng = np.random.RandomState(7)
    return [(int(rng.randint(60, 256)), int(rng.randint(60, 256)),
             int(rng.randint(60, 256))) for _ in range(n)]


def _overlay(frame, labels, probs, current, colors, progress=None) -> None:
    import cv2
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), HEADER_BG, -1)
    text = current
    if progress and progress[1] > 1:
        text += f"   [{progress[0]}/{progress[1]}]"
    cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2,
                cv2.LINE_AA)
    for i, p in enumerate(probs):
        y = 60 + i * 34
        cv2.rectangle(frame, (10, y), (270, y + 26), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, y), (10 + int(260 * float(p)), y + 26),
                      colors[i], -1)
        cv2.putText(frame, f"{labels[i]}  {float(p) * 100:4.1f}%", (16, y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)


def predict_image(mode: str, region: str, image_path: str, *,
                  root: str | Path = artifacts.DEFAULT_ROOT,
                  version: int | str | None = None, draw_landmarks: bool = True,
                  show: bool = True, width: int = 960):
    """Classify a single still image (static models only).

    Returns (label, confidence, probabilities). With show=True it also opens a
    window with the result until a key is pressed.
    """
    import cv2

    predictor = Predictor.load(mode, region, root=root, version=version)
    if predictor.mode != "static":
        raise ValueError("Single-image prediction needs a static model; "
                         f"{mode}/{region} is {predictor.mode}.")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    holistic = make_holistic()
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)
    finally:
        holistic.close()

    vec = predictor.features(res)
    if vec is None:
        print("No landmarks detected in the image.")
        label, conf, probs = None, 0.0, np.zeros(len(predictor.labels), np.float32)
    else:
        probs = predictor.predict(vec)
        top = int(np.argmax(probs))
        label, conf = predictor.labels[top], float(probs[top])
        print(f"{label}   {conf * 100:.1f}%")
        for name, p in zip(predictor.labels, probs):
            print(f"   {name:16} {float(p) * 100:5.1f}%")

    if show:
        h0, w0 = frame.shape[:2]
        if w0 != width:
            frame = cv2.resize(frame, (width, int(h0 * width / w0)))
        if draw_landmarks:
            draw(frame, res, predictor.region)
        colors = _palette(len(predictor.labels))
        _overlay(frame, predictor.labels, probs, label or "-", colors)
        win = f"gesto {mode}/{region} (any key to close)"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return label, conf, probs


def run(mode: str, region: str, *, root: str | Path = artifacts.DEFAULT_ROOT,
        version: int | str | None = None, source: str = "0",
        threshold: float = 0.5, smooth: int = 5, width: int = 960,
        mirror: bool | None = None, draw_landmarks: bool = True) -> None:
    """Open a camera or video and show live predictions."""
    import cv2

    predictor = Predictor.load(mode, region, root=root, version=version)
    labels = predictor.labels
    print(f"{predictor.mode} / {predictor.region}: {labels}")
    print(f"model: {predictor.run}")

    src: int | str = int(source) if str(source).isdigit() else source
    is_webcam = isinstance(src, int)
    if mirror is None:
        mirror = is_webcam          # Gesto mirrors the webcam, not video files

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")

    if is_webcam:
        wait_ms = 1
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_ms = int(1000 / fps) if fps and fps > 1 else 33

    window_name = f"gesto {mode}/{region} (q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    holistic = make_holistic()
    colors = _palette(len(labels))
    recent: deque = deque(maxlen=max(1, smooth))
    current = "-"
    probs = np.zeros(len(labels), np.float32)

    quit_pressed = False
    frame = None
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            h0, w0 = frame.shape[:2]
            if w0 != width:
                frame = cv2.resize(frame, (width, int(h0 * width / w0)))
            if draw_landmarks:
                draw(frame, res, predictor.region)

            vec = predictor.features(res)
            out = predictor.predict(vec)
            if out is not None:
                probs = out
                top = int(np.argmax(probs))
                recent.append(top)
                if (len(recent) == recent.maxlen and len(set(recent)) == 1
                        and float(probs[top]) >= threshold):
                    current = labels[top]
            elif vec is None and predictor.mode == "static":
                recent.clear()
                current = "-"
                probs = np.zeros(len(labels), np.float32)

            _overlay(frame, labels, probs, current, colors,
                     predictor.progress if predictor.mode == "sequence" else None)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                quit_pressed = True
                break

        if not is_webcam and not quit_pressed and frame is not None:
            cv2.putText(frame, "video ended - press any key",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, WHITE, 1, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            cv2.waitKey(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
