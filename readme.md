# Gesto

Collect hand-gesture samples from your webcam, train a model, and run it live.
Gestures can be **any length** — a quick flick or a slow motion — because every
sample is resampled to a fixed length before training. This also makes the model
**speed-invariant**: slow and fast versions of the same gesture look the same.

You collect gestures under numeric labels now, and map those numbers to display
names (letters, words, etc.) at detection time.

## How it works

- Each frame = **63 hand-landmark features** (21 points x x, y, z, one hand).
- Each gesture sample is recorded at its **natural length** (you mark start/stop).
- Before training/detection, every sample is **resampled to 30 frames** by
  interpolation, so the LSTM always sees a fixed `(30, 63)` input.

```
collect (any length)  ->  resample to 30  ->  train (.h5)  ->  detect
```

## Setup

```bash
pip install tensorflow opencv-python mediapipe scikit-learn numpy
```

All scripts use **camera index 0**. Change `CAMERA_INDEX` at the top of a file
if your webcam is elsewhere.

`gesto_common.py` is shared by all scripts (keypoint extraction + resampling) —
keep it in the same folder.

## 1. Collect — `collect_data.py`

Records natural-length samples for one class. Pass the class **number**:

```bash
python collect_data.py 0
python collect_data.py 1
```

- Press **SPACE** to start a sample, **SPACE** again to stop.
- A quick gesture might be 6 frames; a slow one 40 — both are fine.
- **10 samples** per class. Re-running a class **resumes** (no overwrite).
- Press **q** to quit. Samples under 2 frames are discarded.

Stored at raw length:

```
gesture_data/
  0/  0.npy 1.npy ...     # each (L, 63), L varies per sample
  1/  ...
```

Collect at least **2 classes** before training.

## 2. Train — `train.py`

```bash
python train.py
```

Loads all samples, **resamples each to 30 frames**, trains the LSTM, and saves:

- `gesto_model.h5` — the model
- `labels.json` — class index -> the number you collected

Uses an 80/20 stratified split with early stopping and dropout, and prints
validation accuracy.

> 10 samples per class is small for an LSTM, so real-world accuracy may be lower
> than the validation figure. If a class underperforms, collect more for it.

## 3. Detect — `detect.py`

```bash
python detect.py
```

Press-to-detect, mirroring collection:

- Press **SPACE** to start, perform the gesture, **SPACE** to stop.
- The captured frames (any length) are resampled to 30 and classified.
- Predicted name + confidence stays in the header. **q** to quit.

### Map numbers to names

Edit the dict at the top of `detect.py`:

```python
DISPLAY_NAMES = {"0": "A", "1": "B", "2": "C"}
```

Leave empty (`{}`) to show raw numbers.

### Tuning

- `THRESHOLD` (default `0.7`) — below this, the prediction shows as `?`.
- `TARGET_LEN` (in `gesto_common.py`, default `30`) — the fixed length every
  gesture is resampled to. Changing it means you must retrain.

## Files

| File              | Purpose                                       |
|-------------------|-----------------------------------------------|
| `gesto_common.py` | Shared: keypoint extraction + resampling      |
| `collect_data.py` | Capture natural-length samples per class      |
| `train.py`        | Resample to fixed length, train, save         |
| `detect.py`       | Press-to-detect live recognition              |

## Notes

- Uses the **right hand** only, matching how data is collected.
- A 1-frame (static) gesture is handled by repeating the frame to fill 30.
- Because gestures are variable length, both collection and detection are
  **press-to-mark** (SPACE start/stop) rather than a fixed rolling window.

## License

Apache-2.0. Permissive stack throughout — MediaPipe (Apache-2.0), TensorFlow
(Apache-2.0), OpenCV (Apache-2.0), NumPy / scikit-learn (BSD) — usable in
commercial projects without copyleft obligations.
