# Gesto — static (single-frame) detection, all regions

For gestures that are a HELD SHAPE or POSTURE, not a motion. Classifies every
frame on its own — no sequence window, no warm-up, instant response.

| region | dim | train | detect | Gesto project setting |
|---|---|---|---|---|
| hands_one | 63 | `train_static_hands_one.py` | `detect_static_hands_one.py` | Hands, one hand |
| hands_two | 126 | `train_static_hands_two.py` | `detect_static_hands_two.py` | Hands, two hands |
| pose | 132 | `train_static_pose.py` | `detect_static_pose.py` | Pose |
| legs | 32 | `train_static_legs.py` | `detect_static_legs.py` | Legs |
| full | 258 | `train_static_full.py` | `detect_static_full.py` | Full |

Each file is standalone. All open source: MediaPipe (Apache-2.0),
TensorFlow (Apache-2.0), OpenCV (Apache-2.0), NumPy (BSD).

## Capture
In Gesto Labeller, create a project with the matching region and capture in
**Static** mode (not Sequence). Hold each pose and capture ~20-30 frames per
class. Files land in `<project>/data/static/<label>/*.npy`.

## Setup
```bash
pip install tensorflow opencv-python mediapipe numpy
```

## Train & detect (example: pose)
```bash
python train_static_pose.py "D:\...\gesto_projects\my-postures"
python detect_static_pose.py
python detect_static_pose.py artifacts_static_pose --source clip.mp4
```
Train writes `artifacts_static_<region>/`; detect defaults to reading it.

Options: `--threshold 0.6` (min confidence), `--smooth 3` (frames of agreement
before committing; 1 = instant), `--width 1280` (display size). Live probability
bars for every class. Press **q** to quit.

## Static vs LSTM
Use **static** when the gesture IS the pose — hand signs, letters, postures,
stances. Much less data needed (every frame is a sample), instant detection,
and no collapse-to-one-class fragility.

Use the **LSTM** (the sequence bundle) only when two gestures share the same
shape and differ by MOTION — e.g. "waving" vs "hand held up".

## Notes
- Assumes Gesto's "Normalise" was ON (the default). If you captured with it
  OFF, set `NORMALIZE = False` at the top of BOTH files for that region.
- Mirrors the webcam (matching Gesto capture) but not video files.
- Verified: normalization is byte-for-byte identical to Gesto's, and all five
  regions classify real Gesto-captured data at 100%.
