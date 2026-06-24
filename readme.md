# Gesto

A simple gesture-recognition pipeline: collect hand-gesture samples from your
webcam, train a model, and run it live. Built on MediaPipe (hand landmarks) and
a Keras LSTM.

You collect gestures under numeric labels now, and map those numbers to display
names (letters, words, whatever) at detection time.

## How it works

Each gesture sample is a short clip of **30 frames**, and each frame is the
**63 hand-landmark features** MediaPipe gives for one hand (21 points × x, y, z).
The model learns to map a 30-frame sequence to a gesture class.

```
collect  ->  train  ->  detect
 (.npy)      (.h5)      (live)
```

## Setup

```bash
pip install tensorflow opencv-python mediapipe scikit-learn numpy
```

All scripts use **camera index 0**. If your webcam is on a different index,
change `CAMERA_INDEX` at the top of the relevant file.

## 1. Collect data — `collect_data.py`

Records gesture samples for one class at a time. You pass the class **number**:

```bash
python collect_data.py 0      # collect class "0"
python collect_data.py 1      # collect class "1"
```

- **30 frames** per sample, **10 samples** per class
- Press **SPACE** to record the next sample (gives you time to pose your hand)
- Press **q** to quit
- Re-running a class **resumes** from where you left off (won't overwrite)

Samples are saved as one array per sample:

```
gesture_data/
  0/  0.npy 1.npy ... 9.npy     # each file is shape (30, 63)
  1/  0.npy ...
```

Collect at least **2 classes** before training. Keep your hand in frame for the
full 30 frames of each sample.

## 2. Train — `train.py`

```bash
python train.py
```

Loads everything from `gesture_data/`, trains the LSTM, and saves:

- `gesto_model.h5` — the trained model
- `labels.json` — maps class index → the number you collected

It uses an 80/20 train/validation split with early stopping and dropout, and
prints the validation accuracy at the end.

> Note: 10 samples per class is small for an LSTM, so real-world accuracy may be
> lower than the validation number. If a class performs poorly, collect more
> samples for it (`collect_data.py` will resume and add to it).

## 3. Detect — `detect.py`

```bash
python detect.py
```

Opens the webcam, draws the hand landmarks, keeps a rolling **30-frame window**,
and shows the predicted gesture name + confidence on screen. Press **q** to quit.

### Map numbers to names

Edit the dict at the top of `detect.py` to show friendly names instead of the
raw numbers you collected:

```python
DISPLAY_NAMES = {"0": "A", "1": "B", "2": "C"}
```

Leave it empty (`{}`) to display the raw labels.

### Tuning

- `THRESHOLD` (default `0.7`) — predictions below this show as `...` instead of
  a guess. Lower it if detection feels unresponsive; raise it if it's jumpy.

## Files

| File              | Purpose                                  |
|-------------------|------------------------------------------|
| `hand_demo.py`    | Minimal landmark-drawing demo            |
| `collect_data.py` | Capture 30-frame samples per class       |
| `train.py`        | Train the LSTM, save model + labels      |
| `detect.py`       | Live recognition with a sliding window   |

## Notes

- Detection currently uses the **right hand** only, matching how the data is
  collected. A left-hand-only signer would not be recognized.
- Predictions run every frame, so labels can flicker between similar gestures.
  A common fix is to only commit a label once the last few predictions agree.

## License

Apache-2.0. The stack is permissive throughout — MediaPipe (Apache-2.0),
TensorFlow (Apache-2.0), OpenCV (Apache-2.0), NumPy / scikit-learn (BSD) — so it
can be used in commercial projects without copyleft obligations.