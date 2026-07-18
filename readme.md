# Gesto Training & Detection

Train and run gesture models from your Gesto Labeller datasets. Two model
types, each with its own train + detect script:

- **Static** (single frame): each gesture is a held pose. Small Dense network.
- **Sequence** (LSTM): each gesture is a motion over time. Stacked LSTM.

## Files

| File | What it does |
|---|---|
| `gesto_data.py` | Loads a Gesto project's `.npy` data (shared by both trainers) |
| `train_static.py` | Trains the single-frame classifier |
| `train_lstm.py` | Trains the motion (LSTM) classifier |
| `gesto_landmarks.py` | Live landmark extraction (same as the app's capture) |
| `detect_static.py` | Live/video single-frame detection |
| `detect_lstm.py` | Live/video motion detection (rolling window) |
| `convert_tflite.py` | Converts either model to TFLite (LSTM-safe) |

## Setup

```bash
pip install tensorflow opencv-python mediapipe numpy
```

## Get your project path

In Gesto Labeller, click **Copy path** on a project card. That's the
`project_dir` these scripts take.

## Train

```bash
# single-frame model
python train_static.py "D:\products\gesto-labeller\gesto_projects\alphabets"

# motion model
python train_lstm.py "D:\products\gesto-labeller\gesto_projects\motion-hands"
```

Each writes to an `artifacts_*/` folder: the `.keras` model plus `labels.json`
(classes, feature dim, region, and — for LSTM — the sequence length).

Check your data first with:
```bash
python gesto_data.py "path\to\project"
```

## Detect

```bash
python detect_static.py artifacts_static            # webcam
python detect_lstm.py   artifacts_lstm              # webcam
python detect_lstm.py   artifacts_lstm --source clip.mp4 --threshold 0.6
```

Press **q** to quit. Detection uses MediaPipe Holistic — the same engine Gesto
captures with — so the landmarks match what the model was trained on.

## Convert to TFLite (for Flutter etc.)

```bash
python convert_tflite.py artifacts_lstm/lstm_model.keras
```

For LSTM models this clones with `unroll=True` and verifies the TFLite output
matches Keras (plain conversion of LSTM layers can silently be wrong). It prints
a max diff and PASS/WARNING.

## Notes

- **Region must match**: a model trained on Hands data (dim 63/126) can't run on
  Pose data (132). `labels.json` records the region so detection stays consistent.
- **Enough samples**: with fewer than ~5 per class, accuracy is unreliable. The
  trainers warn you.
- **One-hand vs two-hand**: recorded in `labels.json` and applied automatically
  at detection.
