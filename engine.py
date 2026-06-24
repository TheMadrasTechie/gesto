"""Engine — the high-level façade the UI/CLI talk to.

This is the one object a frontend needs. It owns a Dataset, an Extractor, and
(after training) a Classifier, and exposes the verbs of the app:

    define a gesture        -> add_gesture
    capture a sample        -> capture(frame, label)
    train                   -> train
    run live                -> predict(frame)
    save / load project     -> save_project / load_project
    export model            -> export_model

Nothing here imports a UI toolkit, so the same Engine drives PySide6, a CLI,
or a service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .classifier import NearestCentroidClassifier
from .dataset import Dataset
from .extractor import LandmarkExtractor, create_extractor
from .schema import GestureType, LandmarkSource, ProjectMeta


class Engine:
    def __init__(self, meta: ProjectMeta, extractor: Optional[LandmarkExtractor] = None):
        self.dataset = Dataset(meta)
        self.extractor = extractor or create_extractor(meta.source)
        self.classifier: Optional[NearestCentroidClassifier] = None

    # --- factory ----------------------------------------------------------
    @classmethod
    def new_project(
        cls,
        name: str,
        source: LandmarkSource = LandmarkSource.HANDS,
        gesture_type: GestureType = GestureType.STATIC,
    ) -> "Engine":
        meta = ProjectMeta(name=name, source=source, gesture_type=gesture_type)
        return cls(meta)

    # --- gesture definition ----------------------------------------------
    def add_gesture(self, name: str, **kwargs):
        return self.dataset.add_class(name, **kwargs)

    # --- capture ----------------------------------------------------------
    def capture(self, frame: np.ndarray, label: str):
        """Extract landmarks from a frame and store as a labeled sample.

        Returns the Sample, or None if no landmarks were detected.
        """
        feats = self.extractor.extract(frame)
        if feats is None:
            return None
        return self.dataset.add_sample(label, feats)

    def capture_features(self, features, label: str):
        """Store a pre-extracted feature vector (e.g. from an uploaded array)."""
        return self.dataset.add_sample(label, features)

    # --- training ---------------------------------------------------------
    def train(self) -> dict:
        X, y = self.dataset.to_arrays()
        if len(X) == 0:
            raise RuntimeError("no samples to train on")
        if len(set(y)) < 2:
            raise RuntimeError("need at least 2 gesture classes with samples")
        clf = NearestCentroidClassifier()
        report = clf.fit(X, y)
        self.classifier = clf
        return report

    # --- inference --------------------------------------------------------
    def predict(self, frame: np.ndarray) -> Optional[tuple[str, float]]:
        if self.classifier is None:
            raise RuntimeError("model is not trained")
        feats = self.extractor.extract(frame)
        if feats is None:
            return None
        return self.classifier.predict(feats)

    def predict_features(self, features) -> tuple[str, float]:
        if self.classifier is None:
            raise RuntimeError("model is not trained")
        return self.classifier.predict(np.asarray(features, dtype=np.float32))

    # --- persistence ------------------------------------------------------
    def save_project(self, directory: str | Path) -> None:
        self.dataset.save(directory)
        if self.classifier is not None:
            self.classifier.save(Path(directory) / "model")

    @classmethod
    def load_project(cls, directory: str | Path) -> "Engine":
        ds = Dataset.load(directory)
        eng = cls(ds.meta)
        eng.dataset = ds
        model_json = Path(directory) / "model.json"
        if model_json.exists():
            eng.classifier = NearestCentroidClassifier.load(Path(directory) / "model")
        return eng

    def export_model(self, path: str | Path) -> None:
        if self.classifier is None:
            raise RuntimeError("nothing to export — train first")
        self.classifier.save(path)

    def close(self) -> None:
        self.extractor.close()
