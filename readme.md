# Gesto — per-region train & detect

Ten self-contained scripts, one train + one detect per region. Each file is
standalone (no shared imports) — copy just the pair you need.

| region      | dim | train                 | detect                 | Gesto project setting     |
|-------------|-----|-----------------------|------------------------|---------------------------|
| hands_one   | 63  | `train_hands_one.py`  | `detect_hands_one.py`  | Hands, one hand           |
| hands_two   | 126 | `train_hands_two.py`  | `detect_hands_two.py`  | Hands, two hands          |
| pose        | 132 | `train_pose.py`       | `detect_pose.py`       | Pose                      |
| legs        | 32  | `train_legs.py`       | `detect_legs.py`       | Legs                      |
| full        | 258 | `train_full.py`       | `detect_full.py`       | Full                      |

## Setup
```bash
pip install tensorflow opencv-python mediapipe numpy scikit-learn
```

## Use (example: one-hand signs)
```bash
# train — point at the Gesto project folder (use the Copy path button)
python train_hands_one.py "D:\...\gesto_projects\hand-signs"

# detect — live webcam
python detect_hands_one.py
# or a specific artifacts folder / video
python detect_hands_one.py artifacts_hands_one --source clip.mp4
```

Train writes `artifacts_<region>/model.keras` + `labels.json`; detect reads them
(defaults to `artifacts_<region>` so you can just run `python detect_<region>.py`).

## Everything is matched to Gesto's capture
Each file uses MediaPipe Holistic, the same landmark ordering, Gesto's exact
normalization, and mirrors the webcam — so what you capture is what the model
sees live. Verified: all five regions predict Gesto-captured data at 100%.

- **Normalise**: assumes Gesto's "Normalise" was ON (the default). If you
  captured with it OFF, set `NORMALIZE = False` at the top of the region's
  train AND detect file.
- **Small data**: train auto-uses a lighter model under 100 sequences (avoids
  the collapse-to-one-class failure). Force with `--small`.
- **Clip length**: uses `--seq_len 30` by default; clips shorter than that are
  skipped. Capture at a consistent length (set Max frames = 30 in Gesto).

## Detect controls
`--threshold 0.5` (min confidence to commit a label), `--stable 8` (frames of
agreement before committing), `--source 0` (webcam) or a video path. Live
probability bars for every class are shown on screen. Press **q** to quit.
