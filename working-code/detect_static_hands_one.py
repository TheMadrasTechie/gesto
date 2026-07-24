"""
Gesto one hand — STATIC detect (single frame, no motion).

Classifies EVERY FRAME on its own — no 30-frame window, no warm-up, instant
response. Use with a model from train_static_hands_one.py.

Matches Gesto's capture exactly: MediaPipe Holistic, same landmark ordering,
Gesto's normalization (ON by default), and the webcam mirror.

Run:
    python detect_static_hands_one.py
    python detect_static_hands_one.py artifacts_static_hands_one
    python detect_static_hands_one.py artifacts_static_hands_one --source clip.mp4

Press q to quit.
"""

import sys, json, argparse
from collections import deque
from pathlib import Path
import numpy as np

REGION_KEY   = "hands_one"
FEATURE_DIM  = 63
NORMALIZE    = True   # set False if you captured with Gesto "Normalise" unchecked


# ---------- landmark extraction (matches Gesto capture) ----------
def extract(res):
    """One-hand vector: prefer RIGHT hand, else LEFT — exactly like Gesto."""
    hand = res.right_hand_landmarks or res.left_hand_landmarks
    if hand is None:
        return None                      # nothing detected this frame
    return np.array([[r.x, r.y, r.z] for r in hand.landmark],
                    dtype=np.float32).flatten()


# ---------- normalization (EXACT copy of Gesto's normalize_vector) ----------
def _norm_one_hand(pts):
    if np.any(pts):
        pts = pts - pts[0]
        s = np.linalg.norm(pts, axis=1).max()
        if s > 1e-6:
            pts = pts / s
    return pts


def normalize(vec):
    return _norm_one_hand(np.asarray(vec, np.float32).reshape(21, 3).copy()).reshape(-1)


# ---------- drawing ----------
_HAND_BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
               (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),
               (18,19),(19,20),(0,17)]


def draw(frame, res):
    import cv2
    hand = res.right_hand_landmarks or res.left_hand_landmarks
    if not hand:
        return
    h, w = frame.shape[:2]
    pts = [(lm.x, lm.y) for lm in hand.landmark]
    for a, b in _HAND_BONES:
        ax, ay = pts[a]; bx, by = pts[b]
        cv2.line(frame, (int(ax*w), int(ay*h)), (int(bx*w), int(by*h)), (60,180,75), 2)
    for x, y in pts:
        cv2.circle(frame, (int(x*w), int(y*h)), 3, (40,160,230), -1)


def make_holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)


# ---------- detection ----------
import cv2
from tensorflow.keras.models import load_model


def main():
    ap = argparse.ArgumentParser(description="Live static one-hand detection")
    ap.add_argument("artifacts", nargs="?", default="artifacts_static_hands_one",
                    help="Folder with model.keras + labels.json")
    ap.add_argument("--source", default="0", help="Webcam index or video path")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=5,
                    help="Frames of agreement before committing a label (1 = off)")
    ap.add_argument("--width", type=int, default=960, help="Display width")
    args = ap.parse_args()

    art = Path(args.artifacts)
    if not (art / "labels.json").exists():
        sys.exit(f"No labels.json in {art}. Train first with "
                 f"train_static_hands_one.py")
    meta = json.loads((art / "labels.json").read_text())
    labels = meta["labels"]
    use_norm = meta.get("normalized", True)
    model = load_model(art / "model.keras")
    print(f"static hands_one model: {labels} (normalize={use_norm})")

    holistic = make_holistic()
    src = int(args.source) if args.source.isdigit() else args.source
    is_webcam = isinstance(src, int)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {src}")
    # pace video to its real frame rate; webcam stays snappy
    if is_webcam:
        wait_ms = 1
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_ms = int(1000 / fps) if fps and fps > 1 else 33

    win = "Gesto static hands_one detect (q=quit)"
    DISPLAY_W = args.width
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    rng = np.random.RandomState(7)
    colors = [(int(rng.randint(60,256)), int(rng.randint(60,256)),
               int(rng.randint(60,256))) for _ in labels]

    recent = deque(maxlen=max(1, args.smooth))
    current = "-"
    probs = np.zeros(len(labels), np.float32)

    quit_pressed = False
    frame = None
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if is_webcam:
            frame = cv2.flip(frame, 1)          # mirror, matching Gesto capture
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
        res = holistic.process(rgb)

        # scale for display so the whole frame is visible
        h0, w0 = frame.shape[:2]
        if w0 != DISPLAY_W:
            frame = cv2.resize(frame, (DISPLAY_W, int(h0 * DISPLAY_W / w0)))
        draw(frame, res)

        kp = extract(res)
        if kp is not None:
            if use_norm:
                kp = normalize(kp)
            # classify THIS frame on its own — no sequence buffer, instant result
            probs = model.predict(kp[None, :], verbose=0)[0]
            top = int(np.argmax(probs))
            recent.append(top)
            if (len(recent) == recent.maxlen and len(set(recent)) == 1
                    and float(probs[top]) > args.threshold):
                current = labels[top]
        else:
            recent.clear()
            current = "-"
            probs = np.zeros(len(labels), np.float32)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (245, 117, 16), -1)
        hint = "" if kp is not None else "  (no hand)"
        cv2.putText(frame, f"{current}{hint}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        for i, p in enumerate(probs):
            yb = 60 + i * 34
            cv2.rectangle(frame, (10, yb), (270, yb + 26), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, yb), (10 + int(260*float(p)), yb + 26), colors[i], -1)
            cv2.putText(frame, f"{labels[i]}  {float(p)*100:4.1f}%", (16, yb + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(win, frame)
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            quit_pressed = True
            break

    if not is_webcam and not quit_pressed and frame is not None:
        cv2.putText(frame, "video ended - press any key", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, frame)
        cv2.waitKey(0)

    cap.release(); cv2.destroyAllWindows(); holistic.close()


if __name__ == "__main__":
    main()
