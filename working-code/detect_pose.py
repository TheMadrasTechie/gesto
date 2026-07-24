"""
Gesto full-body pose — detect.

Self-contained detect script for the pose region (dim 132).
Reads Gesto Labeller data captured with region="Pose".
No other project files needed — just this file.

Matches Gesto's capture exactly: MediaPipe Holistic, the same landmark ordering,
the same normalization (Gesto's "Normalise" is ON by default), and the webcam
mirror. If you captured with Normalise UNCHECKED, set NORMALIZE = False below.
"""

import os, sys, json, argparse
from pathlib import Path
from collections import deque
import numpy as np

REGION_KEY   = "pose"
GESTO_REGION = "Pose"
HANDS        = "two"
FEATURE_DIM  = 132
NORMALIZE    = True   # set False if you captured with Gesto "Normalise" unchecked

LEG_POSE_IDX = [23, 24, 25, 26, 27, 28, 31, 32]


# ---------- landmark extraction (matches Gesto capture) ----------
def _rh(res):
    return (np.array([[r.x, r.y, r.z] for r in res.right_hand_landmarks.landmark]).flatten()
            if res.right_hand_landmarks else np.zeros(63, np.float32))

def _lh(res):
    return (np.array([[r.x, r.y, r.z] for r in res.left_hand_landmarks.landmark]).flatten()
            if res.left_hand_landmarks else np.zeros(63, np.float32))

def _pose_full(res):
    return (np.array([[r.x, r.y, r.z, r.visibility] for r in res.pose_landmarks.landmark]).flatten()
            if res.pose_landmarks else np.zeros(132, np.float32))

def _pose_legs(res):
    if not res.pose_landmarks:
        return np.zeros(len(LEG_POSE_IDX) * 4, np.float32)
    lm = res.pose_landmarks.landmark
    return np.array([[lm[i].x, lm[i].y, lm[i].z, lm[i].visibility] for i in LEG_POSE_IDX]).flatten()

def extract(res):
    """Return the pose feature vector (raw, unnormalized)."""
    return _pose_full(res).astype(np.float32)


# ---------- normalization (EXACT copy of Gesto's normalize_vector) ----------
def _norm_one_hand(pts):
    if np.any(pts):
        pts = pts - pts[0]
        s = np.linalg.norm(pts, axis=1).max()
        if s > 1e-6:
            pts = pts / s
    return pts

def normalize(vec):
    vec = np.asarray(vec, np.float32)
    if REGION_KEY == "hands_one":
        return _norm_one_hand(vec.reshape(21, 3).copy()).reshape(-1)
    if REGION_KEY == "hands_two":
        pts = vec.reshape(2, 21, 3).copy()
        for h in range(2):
            pts[h] = _norm_one_hand(pts[h])
        return pts.reshape(-1)
    if REGION_KEY in ("pose", "legs"):
        n = 33 if REGION_KEY == "pose" else len(LEG_POSE_IDX)
        pts = vec.reshape(n, 4).copy()
        xyz = pts[:, :3]
        mask = np.any(xyz != 0, axis=1)
        if np.any(mask):
            xyz -= xyz[mask].mean(axis=0)
            s = np.linalg.norm(xyz, axis=1).max()
            if s > 1e-6:
                xyz /= s
        pts[:, :3] = xyz
        return pts.reshape(-1)
    if REGION_KEY == "full":
        # pose(132) normalized as pose, hands(126) normalized per-hand
        pose = vec[:132].reshape(33, 4).copy()
        xyz = pose[:, :3]; mask = np.any(xyz != 0, axis=1)
        if np.any(mask):
            xyz -= xyz[mask].mean(axis=0)
            s = np.linalg.norm(xyz, axis=1).max()
            if s > 1e-6: xyz /= s
        pose[:, :3] = xyz
        hands = vec[132:].reshape(2, 21, 3).copy()
        for h in range(2):
            hands[h] = _norm_one_hand(hands[h])
        return np.concatenate([pose.reshape(-1), hands.reshape(-1)])
    return vec


# ---------- drawing (region-aware) ----------
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
                cv2.line(frame, (int(ax*w),int(ay*h)), (int(bx*w),int(by*h)), (60,180,75), 2)
    for x, y in pts:
        if x or y:
            cv2.circle(frame, (int(x*w),int(y*h)), 3, (40,160,230), -1)

def draw(frame, res):
    hx = lambda h: [(lm.x, lm.y) for lm in h.landmark] if h else []
    if REGION_KEY == "hands_one":
        hand = res.right_hand_landmarks or res.left_hand_landmarks
        if hand: _dots(frame, hx(hand), _HAND_BONES)
    elif REGION_KEY == "hands_two":
        for h in (res.left_hand_landmarks, res.right_hand_landmarks):
            if h: _dots(frame, hx(h), _HAND_BONES)
    elif REGION_KEY == "pose" and res.pose_landmarks:
        _dots(frame, [(lm.x,lm.y) for lm in res.pose_landmarks.landmark], _POSE_BONES)
    elif REGION_KEY == "legs" and res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        _dots(frame, [(lm[i].x, lm[i].y) for i in LEG_POSE_IDX], _LEG_BONES)
    elif REGION_KEY == "full":
        if res.pose_landmarks:
            _dots(frame, [(lm.x,lm.y) for lm in res.pose_landmarks.landmark], _POSE_BONES)
        for h in (res.left_hand_landmarks, res.right_hand_landmarks):
            if h: _dots(frame, hx(h), _HAND_BONES)

def make_holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)


# ---------- detection ----------
import cv2
from tensorflow.keras.models import load_model


def main():
    ap = argparse.ArgumentParser(description="Live pose gesture detection")
    ap.add_argument("artifacts", nargs="?", default="artifacts_pose",
                    help="Folder with model.keras + labels.json")
    ap.add_argument("--source", default="0", help="Webcam index or video path")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stable", type=int, default=8)
    args = ap.parse_args()

    art = Path(args.artifacts)
    meta = json.loads((art / "labels.json").read_text())
    labels = meta["labels"]; T = meta["seq_len"]
    use_norm = meta.get("normalized", True)
    model = load_model(art / "model.keras")
    print(f"pose model: {labels} (T={T}, normalize={use_norm})")

    holistic = make_holistic()
    src = int(args.source) if args.source.isdigit() else args.source
    is_webcam = isinstance(src, int)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {src}")
    # pace video playback to its real frame rate; webcam stays snappy at ~1ms
    if is_webcam:
        wait_ms = 1
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_ms = int(1000 / fps) if fps and fps > 1 else 33   # default ~30fps
    win = "Gesto pose detect (q=quit)"
    DISPLAY_W = 960          # frames are scaled to this width for display
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    rng = np.random.RandomState(7)
    colors = [(int(rng.randint(60,256)), int(rng.randint(60,256)), int(rng.randint(60,256)))
              for _ in labels]
    sequence = []; preds = deque(maxlen=args.stable); current = "-"
    probs = np.zeros(len(labels), np.float32)

    quit_pressed = False
    frame = None
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break                       # end of video / stream
        if is_webcam:
            frame = cv2.flip(frame, 1)          # mirror, matching Gesto capture
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
        res = holistic.process(rgb)

        # scale the frame to a consistent display width so the WHOLE frame is
        # visible (large source videos were opening zoomed-in / cropped)
        h0, w0 = frame.shape[:2]
        if w0 != DISPLAY_W:
            frame = cv2.resize(frame, (DISPLAY_W, int(h0 * DISPLAY_W / w0)))
        draw(frame, res)

        kp = extract(res)
        if use_norm:
            kp = normalize(kp)
        sequence.append(kp); sequence = sequence[-T:]

        if len(sequence) == T:
            probs = model.predict(np.expand_dims(sequence, 0), verbose=0)[0]
            top = int(np.argmax(probs)); preds.append(top)
            if (len(preds) == preds.maxlen and len(set(preds)) == 1
                    and float(probs[top]) > args.threshold):
                current = labels[top]

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (245, 117, 16), -1)
        cv2.putText(frame, f"{current}   [{len(sequence)}/{T}]", (10, 32),
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

    # if the video finished on its own (not a manual quit), hold the last frame
    # so the final result stays on screen until you press a key
    if not is_webcam and not quit_pressed and frame is not None:
        cv2.putText(frame, "video ended - press any key", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, frame)
        cv2.waitKey(0)

    cap.release(); cv2.destroyAllWindows(); holistic.close()


if __name__ == "__main__":
    main()
