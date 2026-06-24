"""Core data types and schema for Gesto.

These are plain dataclasses with no UI or ML-framework dependencies, so they
can be imported anywhere. The on-disk format is JSON for metadata plus a
compact array store for the landmark data itself.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


SCHEMA_VERSION = 1


class LandmarkSource(str, Enum):
    """Which MediaPipe model feeds landmarks into the recognizer.

    The feature-vector length per frame depends on this choice:
      HANDS     21 pts x 3 (x, y, z)               =   63   (one hand)
      HANDS_2   21 pts x 3 x 2 hands               =  126
      POSE      33 pts x 4 (x, y, z, visibility)   =  132
      HOLISTIC  pose(33x4) + 2 hands(21x3 each)    =  258
    """

    HANDS = "hands"
    HANDS_2 = "hands_2"
    POSE = "pose"
    HOLISTIC = "holistic"

    @property
    def feature_dim(self) -> int:
        return {
            LandmarkSource.HANDS: 21 * 3,
            LandmarkSource.HANDS_2: 21 * 3 * 2,
            LandmarkSource.POSE: 33 * 4,
            LandmarkSource.HOLISTIC: 33 * 4 + 21 * 3 * 2,
        }[self]


class GestureType(str, Enum):
    """v1 supports STATIC only. DYNAMIC is reserved for a later version."""

    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass
class Sample:
    """A single captured example of a gesture.

    For STATIC gestures, `features` is one frame: shape (feature_dim,).
    For DYNAMIC (future), it will be a sequence: shape (frames, feature_dim).
    Stored as a plain list here; the array store handles efficient packing.
    """

    label: str
    features: list[float]
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(**d)


@dataclass
class GestureClass:
    """One gesture the user has defined (e.g. 'thumbs_up')."""

    name: str
    display_name: Optional[str] = None
    note: str = ""

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.name.replace("_", " ").title()


@dataclass
class ProjectMeta:
    """Top-level project metadata. The actual samples live in the store."""

    name: str
    source: LandmarkSource
    gesture_type: GestureType = GestureType.STATIC
    schema_version: int = SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)
    mediapipe_version: str = ""
    description: str = ""

    @property
    def feature_dim(self) -> int:
        return self.source.feature_dim

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        d["gesture_type"] = self.gesture_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectMeta":
        d = dict(d)
        d["source"] = LandmarkSource(d["source"])
        d["gesture_type"] = GestureType(d.get("gesture_type", "static"))
        return cls(**d)
