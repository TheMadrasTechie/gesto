# Gesto LSTM (notebook-style) — per region

LSTM training + live detection that matches your `Alphabet_recognition.ipynb`:
fixed-length sequences (no zero-padding), your exact architecture
(64→128→64 LSTM + Dense 64→32→softmax, relu), and the rolling `sequence[-T:]`
detection window with a stability check.

Works for all five region types — pass `--region`:

| region        | dim | what it captures            |
|---------------|-----|-----------------------------|
| `hands_right` | 63  | right hand only (notebook)  |
| `hands_two`   | 126 | both hands                  |
| `pose`        | 132 | full body                   |
| `legs`        | 32  | lower body                  |
| `full`        | 258 | body + both hands           |

## Setup
```bash
pip install tensorflow opencv-python mediapipe numpy
```

## Train
Point at a Gesto project folder (the **Copy path** button gives you this):
```bash
python train.py "D:\...\gesto_projects\alphabets" --region hands_right
python train.py "D:\...\gesto_projects\motion-hands" --region hands_two --epochs 400
python train.py "D:\...\gesto_projects\walk-proj" --region pose
```
Writes `artifacts_<region>/model.keras` + `labels.json`.

Only sequences with at least `--seq_len` frames (default 30) are used; longer
ones are trimmed to the first `seq_len`. This matches the notebook, where every
video was exactly 30 frames — so **capture your gestures at a consistent length
(~30 frames)** for best results.

## Detect
```bash
python detect.py artifacts_hands_right
python detect.py artifacts_pose --source clip.mp4 --threshold 0.6
```
Rolling window of the last `seq_len` frames, predicts once full, and only
accepts a label once the last `--stable` (default 10) predictions agree — the
same logic as your notebook. Press **q** to quit.

## Notes
- **Region must match the project.** `train.py` checks the `.npy` dimension and
  stops if you pass the wrong `--region`.
- This is the fixed-length approach from your notebook. It differs from the
  earlier padded scripts — for the alphabet/fingerspelling style where every
  clip is the same length, this is the right one.
- Convert to TFLite with the `convert_tflite.py` from the earlier bundle (the
  `unroll=True` version) — it works on these models unchanged.
