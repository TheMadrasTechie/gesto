"""
Gesto Labeller
==============
A labelling studio for hand-gesture training data.

- Sources: live webcam OR recorded video files
- Modes:   static (single-frame poses) and sequence (motion over time)
- Output:  MediaPipe landmark data as .npy (LSTM-ready) and/or .csv

Stack: PySide6 (LGPL, commercial-safe), MediaPipe, OpenCV, NumPy.
Designed to plug into Gesto's existing gesto_common.py resampling.

Layout written to:
    <out_dir>/
        labels.json                 # label registry + metadata
        static/<label>/<uid>.npy    # (63,) flattened single-frame landmarks
        sequence/<label>/<uid>.npy  # (T, 63) variable-length sequences
        manifest.csv                # every sample: label, mode, frames, path

Run:
    pip install PySide6 mediapipe opencv-python numpy
    python gesto_labeller.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("mediapipe is required: pip install mediapipe")
    raise

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QLineEdit, QFileDialog, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox, QRadioButton, QButtonGroup,
    QStatusBar, QMessageBox, QSpinBox, QCheckBox,
)

NUM_LANDMARKS = 21          # MediaPipe Hands
COORDS = 3                  # x, y, z
FLAT = NUM_LANDMARKS * COORDS  # 63


# --------------------------------------------------------------------------- #
# Landmark extraction
# --------------------------------------------------------------------------- #
def landmarks_to_vector(hand_landmarks) -> np.ndarray:
    """Flatten a MediaPipe hand result into a (63,) float32 vector."""
    vec = np.empty(FLAT, dtype=np.float32)
    for i, lm in enumerate(hand_landmarks.landmark):
        vec[i * 3 + 0] = lm.x
        vec[i * 3 + 1] = lm.y
        vec[i * 3 + 2] = lm.z
    return vec


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Wrist-relative + scale-normalised, so labels are translation/scale invariant."""
    pts = vec.reshape(NUM_LANDMARKS, COORDS).copy()
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-6:
        pts /= scale
    return pts.reshape(-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Video capture worker (threaded so the UI stays responsive)
# --------------------------------------------------------------------------- #
class CaptureWorker(QThread):
    frame_ready = Signal(object, object)   # (bgr_frame, vector|None)
    finished_source = Signal()

    def __init__(self, source=0, loop_video=False, parent=None):
        super().__init__(parent)
        self.source = source
        self.loop_video = loop_video
        self._running = True
        self._hands = None

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.finished_source.emit()
            return

        is_file = not isinstance(self.source, int)
        delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                if is_file and self.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = self._hands.process(rgb)

            vec = None
            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                vec = landmarks_to_vector(hl)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hl, mp.solutions.hands.HAND_CONNECTIONS
                )

            self.frame_ready.emit(frame, vec)
            if is_file and delay:
                time.sleep(delay)

        cap.release()
        self._hands.close()
        self.finished_source.emit()


# --------------------------------------------------------------------------- #
# Dataset on disk
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    out_dir: Path
    normalize: bool = True
    labels: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        (self.out_dir / "static").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "sequence").mkdir(parents=True, exist_ok=True)
        self._load_labels()

    @property
    def _labels_file(self) -> Path:
        return self.out_dir / "labels.json"

    @property
    def _manifest_file(self) -> Path:
        return self.out_dir / "manifest.csv"

    def _load_labels(self):
        if self._labels_file.exists():
            self.labels = json.loads(self._labels_file.read_text()).get("labels", [])

    def _save_labels(self):
        self._labels_file.write_text(json.dumps({"labels": self.labels}, indent=2))

    def add_label(self, name: str) -> bool:
        name = name.strip()
        if not name or name in self.labels:
            return False
        self.labels.append(name)
        (self.out_dir / "static" / name).mkdir(exist_ok=True)
        (self.out_dir / "sequence" / name).mkdir(exist_ok=True)
        self._save_labels()
        return True

    def _append_manifest(self, label, mode, frames, path):
        new = not self._manifest_file.exists()
        with self._manifest_file.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["uid", "label", "mode", "frames", "path"])
            w.writerow([path.stem, label, mode, frames, str(path)])

    def save_static(self, label: str, vec: np.ndarray) -> Path:
        if self.normalize:
            vec = normalize_vector(vec)
        uid = uuid.uuid4().hex[:12]
        path = self.out_dir / "static" / label / f"{uid}.npy"
        np.save(path, vec)
        self._append_manifest(label, "static", 1, path)
        return path

    def save_sequence(self, label: str, seq: list[np.ndarray]) -> Path:
        arr = np.stack([normalize_vector(v) if self.normalize else v for v in seq])
        uid = uuid.uuid4().hex[:12]
        path = self.out_dir / "sequence" / label / f"{uid}.npy"
        np.save(path, arr.astype(np.float32))
        self._append_manifest(label, "sequence", len(seq), path)
        return path

    def count(self, label: str, mode: str) -> int:
        d = self.out_dir / mode / label
        return len(list(d.glob("*.npy"))) if d.exists() else 0


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class GestoLabeller(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesto Labeller")
        self.resize(1100, 720)

        self.dataset: Dataset | None = None
        self.worker: CaptureWorker | None = None
        self.last_vector: np.ndarray | None = None
        self.recording = False
        self.seq_buffer: list[np.ndarray] = []

        self._build_ui()
        self._build_shortcuts()
        self.statusBar().showMessage("Pick an output folder to begin.")

    # ---- UI ---------------------------------------------------------------- #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # --- left: video ---
        left = QVBoxLayout()
        self.video = QLabel("No source")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background:#111;color:#777;border-radius:8px;")
        left.addWidget(self.video, 1)

        src = QHBoxLayout()
        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_webcam.clicked.connect(self.start_webcam)
        self.btn_video = QPushButton("Open Video…")
        self.btn_video.clicked.connect(self.open_video)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_source)
        self.btn_stop.setEnabled(False)
        src.addWidget(self.btn_webcam)
        src.addWidget(self.btn_video)
        src.addWidget(self.btn_stop)
        left.addLayout(src)
        root.addLayout(left, 2)

        # --- right: controls ---
        right = QVBoxLayout()

        # output folder
        out_box = QGroupBox("Output")
        ob = QVBoxLayout(out_box)
        self.btn_out = QPushButton("Choose Folder…")
        self.btn_out.clicked.connect(self.choose_output)
        self.lbl_out = QLabel("—")
        self.lbl_out.setWordWrap(True)
        self.chk_norm = QCheckBox("Normalise (wrist-relative + scaled)")
        self.chk_norm.setChecked(True)
        ob.addWidget(self.btn_out)
        ob.addWidget(self.lbl_out)
        ob.addWidget(self.chk_norm)
        right.addWidget(out_box)

        # mode
        mode_box = QGroupBox("Mode")
        mb = QHBoxLayout(mode_box)
        self.rb_static = QRadioButton("Static pose")
        self.rb_seq = QRadioButton("Sequence")
        self.rb_static.setChecked(True)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.rb_static)
        self.mode_group.addButton(self.rb_seq)
        mb.addWidget(self.rb_static)
        mb.addWidget(self.rb_seq)
        right.addWidget(mode_box)

        # labels
        lbl_box = QGroupBox("Labels")
        lb = QVBoxLayout(lbl_box)
        add_row = QHBoxLayout()
        self.in_label = QLineEdit()
        self.in_label.setPlaceholderText("New label name")
        self.in_label.returnPressed.connect(self.add_label)
        self.btn_add = QPushButton("Add")
        self.btn_add.clicked.connect(self.add_label)
        add_row.addWidget(self.in_label)
        add_row.addWidget(self.btn_add)
        lb.addLayout(add_row)
        self.label_list = QListWidget()
        lb.addWidget(self.label_list)
        right.addWidget(lbl_box, 1)

        # capture
        cap_box = QGroupBox("Capture")
        cb = QVBoxLayout(cap_box)
        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("Max seq frames:"))
        self.spin_max = QSpinBox()
        self.spin_max.setRange(5, 300)
        self.spin_max.setValue(60)
        seq_row.addWidget(self.spin_max)
        cb.addLayout(seq_row)
        self.btn_capture = QPushButton("Capture  (Space)")
        self.btn_capture.clicked.connect(self.on_capture)
        self.btn_capture.setEnabled(False)
        cb.addWidget(self.btn_capture)
        self.lbl_rec = QLabel("")
        self.lbl_rec.setStyleSheet("color:#e44;")
        cb.addWidget(self.lbl_rec)
        right.addWidget(cap_box)

        root.addLayout(right, 1)

    def _build_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self.on_capture)

    # ---- output / labels --------------------------------------------------- #
    @Slot()
    def choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not d:
            return
        self.dataset = Dataset(d, normalize=self.chk_norm.isChecked())
        self.lbl_out.setText(d)
        self.refresh_labels()
        self.statusBar().showMessage(f"Dataset ready: {d}")

    @Slot()
    def add_label(self):
        if not self.dataset:
            QMessageBox.warning(self, "No dataset", "Choose an output folder first.")
            return
        if self.dataset.add_label(self.in_label.text()):
            self.in_label.clear()
            self.refresh_labels()

    def refresh_labels(self):
        self.label_list.clear()
        if not self.dataset:
            return
        mode = "static" if self.rb_static.isChecked() else "sequence"
        for name in self.dataset.labels:
            n = self.dataset.count(name, mode)
            item = QListWidgetItem(f"{name}   ({n})")
            item.setData(Qt.UserRole, name)
            self.label_list.addItem(item)
        if self.label_list.count() and not self.label_list.currentItem():
            self.label_list.setCurrentRow(0)

    def current_label(self) -> str | None:
        it = self.label_list.currentItem()
        return it.data(Qt.UserRole) if it else None

    # ---- sources ----------------------------------------------------------- #
    def start_webcam(self):
        self._start_worker(0)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "", "Video (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self._start_worker(path, loop=True)

    def _start_worker(self, source, loop=False):
        self.stop_source()
        self.worker = CaptureWorker(source, loop_video=loop)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.finished_source.connect(self.on_source_end)
        self.worker.start()
        self.btn_stop.setEnabled(True)
        self.btn_capture.setEnabled(True)

    def stop_source(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.recording = False
        self.seq_buffer.clear()
        self.lbl_rec.setText("")
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)

    @Slot()
    def on_source_end(self):
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.statusBar().showMessage("Source ended.")

    # ---- frames ------------------------------------------------------------ #
    @Slot(object, object)
    def on_frame(self, frame, vec):
        self.last_vector = vec

        if self.recording and vec is not None:
            self.seq_buffer.append(vec)
            self.lbl_rec.setText(f"● REC  {len(self.seq_buffer)} frames")
            if len(self.seq_buffer) >= self.spin_max.value():
                self._finish_sequence()

        h, w = frame.shape[:2]
        img = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
        self.video.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # ---- capture ----------------------------------------------------------- #
    @Slot()
    def on_capture(self):
        if not self.dataset or not self.worker:
            return
        label = self.current_label()
        if not label:
            self.statusBar().showMessage("Select or add a label first.")
            return
        self.dataset.normalize = self.chk_norm.isChecked()

        if self.rb_static.isChecked():
            if self.last_vector is None:
                self.statusBar().showMessage("No hand detected.")
                return
            p = self.dataset.save_static(label, self.last_vector)
            self.statusBar().showMessage(f"Saved static → {p.name}")
            self.refresh_labels()
        else:
            if not self.recording:
                self.recording = True
                self.seq_buffer = []
                self.lbl_rec.setText("● REC  0 frames")
            else:
                self._finish_sequence()

    def _finish_sequence(self):
        self.recording = False
        label = self.current_label()
        if label and len(self.seq_buffer) >= 2:
            p = self.dataset.save_sequence(label, self.seq_buffer)
            self.statusBar().showMessage(
                f"Saved sequence ({len(self.seq_buffer)} frames) → {p.name}"
            )
        else:
            self.statusBar().showMessage("Sequence too short — discarded.")
        self.seq_buffer.clear()
        self.lbl_rec.setText("")
        self.refresh_labels()

    # ---- lifecycle --------------------------------------------------------- #
    def closeEvent(self, e):
        self.stop_source()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    w = GestoLabeller()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()