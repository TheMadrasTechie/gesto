"""Gesture classifier.

v1 targets STATIC gestures, so a single-frame feature vector maps to one class.
The default model is a small, dependency-light nearest-centroid classifier so
the whole pipeline trains and predicts with only numpy. A stronger backend
(Keras dense net, or the user's LSTM for the future dynamic mode) can implement
the same interface and be swapped in without changing callers.

Interface:
    fit(X, y)        -> train
    predict(x)       -> (label, confidence)
    save(path) / load(path)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .normalize import normalize_vector


class Classifier(Protocol):
    labels: list[str]

    def fit(self, X: np.ndarray, y: list[str]) -> dict: ...
    def predict(self, x: np.ndarray) -> tuple[str, float]: ...
    def save(self, path: str | Path) -> None: ...


class NearestCentroidClassifier:
    """Baseline static-gesture classifier.

    Stores the normalized mean vector (centroid) per class. Prediction picks
    the nearest centroid by cosine similarity, mapped to a 0..1 confidence.
    Fast, transparent, and works from a handful of samples — ideal as the v1
    default and as a fallback.
    """

    def __init__(self):
        self.labels: list[str] = []
        self._centroids: np.ndarray | None = None  # (n_classes, feature_dim)

    def fit(self, X: np.ndarray, y: list[str]) -> dict:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("need a non-empty 2D feature matrix to train")
        Xn = np.stack([normalize_vector(row) for row in X])
        self.labels = sorted(set(y))
        cents = []
        for label in self.labels:
            mask = np.array([yi == label for yi in y])
            cents.append(Xn[mask].mean(axis=0))
        self._centroids = np.stack(cents).astype(np.float32)
        # crude train accuracy for a quick sanity signal
        correct = sum(self.predict(X[i])[0] == y[i] for i in range(len(X)))
        return {
            "n_samples": int(len(X)),
            "n_classes": len(self.labels),
            "train_accuracy": round(correct / len(X), 4),
        }

    def predict(self, x: np.ndarray) -> tuple[str, float]:
        if self._centroids is None:
            raise RuntimeError("classifier is not trained")
        xn = normalize_vector(np.asarray(x, dtype=np.float32).ravel())
        # Euclidean distance to each centroid (in normalized space)
        dists = np.linalg.norm(self._centroids - xn, axis=1)
        idx = int(np.argmin(dists))
        # softmax over negative distances -> calibrated-ish confidence
        logits = -dists
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return self.labels[idx], round(float(probs[idx]), 4)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p.with_suffix(".npz"), centroids=self._centroids
        )
        p.with_suffix(".json").write_text(
            json.dumps({"labels": self.labels, "type": "nearest_centroid"}, indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "NearestCentroidClassifier":
        p = Path(path)
        clf = cls()
        meta = json.loads(p.with_suffix(".json").read_text())
        clf.labels = meta["labels"]
        clf._centroids = np.load(p.with_suffix(".npz"))["centroids"]
        return clf
