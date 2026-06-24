# Gesto

A desktop studio for training your own gesture recognizers. Define the gestures
you want, capture a handful of examples, train a model, and run it live — no
predefined vocabulary, bring whatever gestures you need.

## Status

v1 scope: **static gestures** (held poses). Landmark source is selectable —
hands only, full body pose, or holistic. Dynamic (motion) gestures are planned
for a later version and the schema already reserves space for them.

## Architecture

The guiding rule: **the core is UI-agnostic**. Everything in `gesto/core/`
imports no GUI toolkit, so the same engine drives a desktop app, a CLI, or a
service.

```
gesto/
  core/                 pure-Python engine (no UI imports)
    schema.py           data types: LandmarkSource, Sample, ProjectMeta ...
    extractor.py        landmark extraction (MediaPipe, with a stub fallback)
    normalize.py        translation/scale normalization of landmarks
    dataset.py          gesture classes + samples, save/load to disk
    classifier.py       static-gesture model (nearest-centroid baseline)
    engine.py           Engine — the façade the frontends talk to
  cli/                  thin CLI over the engine
  tests/                core smoke tests
```

The single object a frontend needs is `gesto.core.Engine`:

```python
from gesto.core import Engine, LandmarkSource

eng = Engine.new_project("hand_signs", source=LandmarkSource.HANDS)
eng.add_gesture("thumbs_up")
eng.add_gesture("stop")
# eng.capture(frame, "thumbs_up")   # frame from a webcam
report = eng.train()
label, confidence = eng.predict(frame)
eng.save_project("my_project/")
```

## Install

```bash
pip install -e .            # core only (numpy)
pip install -e ".[mediapipe]"   # add real landmark extraction
pip install -e ".[gui]"     # add the PySide6 desktop UI (later)
pip install -e ".[dev]"     # tests
```

## Try it now

```bash
gesto info        # version + landmark sources
gesto demo        # synthetic end-to-end run (no camera needed)
```

## License

Apache-2.0. The stack is deliberately permissive throughout — MediaPipe
(Apache-2.0), NumPy (BSD), PySide6 (LGPL) — so Gesto can be shipped in
commercial products without copyleft obligations.
