"""
Gesto Labeller
==============
A labelling studio for gesture / action training data.

- Sources: live webcam OR recorded video files
- Regions: hands only, pose (whole body), legs only, or full (pose + both hands)
- Modes:   static (single-frame poses) and sequence (motion over time)
- Output:  MediaPipe landmark data as .npy (LSTM-ready) + manifest.csv

Stack: PySide6 (LGPL, commercial-safe), MediaPipe Holistic, OpenCV, NumPy.
Designed to plug into Gesto's existing gesto_common.py resampling.

Layout written to:
    <out_dir>/
        labels.json                        # label registry
        static/<region>/<label>/<uid>.npy  # (D,) single-frame
        sequence/<region>/<label>/<uid>.npy# (T, D) variable-length
        manifest.csv                       # every sample: label, region, mode, frames, dim, path

Feature dimension D depends on region:
    hands  -> 2 hands x 21 x 3 = 126
    pose   -> 33 x 4           = 132   (x, y, z, visibility)
    legs   -> 8  x 4           =  32   (hips/knees/ankles/heels/foot-index)
    full   -> 132 + 126        = 258

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

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QLineEdit, QFileDialog, QVBoxLayout,
    QHBoxLayout, QGroupBox, QRadioButton, QButtonGroup,
    QMessageBox, QSpinBox, QCheckBox,
)

# --------------------------------------------------------------------------- #
# Region definitions
# --------------------------------------------------------------------------- #
# MediaPipe Pose landmark indices for the "legs" subset.
LEG_POSE_IDX = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # hips->feet
# (left/right hip, knee, ankle, heel, foot_index)

REGIONS = {
    "hands": {"dim": 2 * 21 * 3, "label": "Hands only"},
    "pose":  {"dim": 33 * 4,     "label": "Pose (whole body)"},
    "legs":  {"dim": len(LEG_POSE_IDX) * 4, "label": "Legs only"},
    "full":  {"dim": 33 * 4 + 2 * 21 * 3,   "label": "Full (body + hands)"},
}


# --------------------------------------------------------------------------- #
# Landmark extraction from a Holistic result
# --------------------------------------------------------------------------- #
def _pose_array(res, indices=None) -> np.ndarray:
    """Pose landmarks as (N,4) x,y,z,visibility. Zeros if absent."""
    n = len(indices) if indices is not None else 33
    out = np.zeros((n, 4), dtype=np.float32)
    if res.pose_landmarks:
        lms = res.pose_landmarks.landmark
        idxs = indices if indices is not None else range(33)
        for j, i in enumerate(idxs):
            lm = lms[i]
            out[j] = (lm.x, lm.y, lm.z, lm.visibility)
    return out.reshape(-1)


def _hand_array(hand_landmarks) -> np.ndarray:
    """One hand as (21,3) x,y,z. Zeros if absent."""
    out = np.zeros((21, 3), dtype=np.float32)
    if hand_landmarks:
        for i, lm in enumerate(hand_landmarks.landmark):
            out[i] = (lm.x, lm.y, lm.z)
    return out.reshape(-1)


def extract_vector(res, region: str) -> np.ndarray | None:
    """Build the feature vector for the chosen region. None if nothing detected."""
    if region == "hands":
        left = res.left_hand_landmarks
        right = res.right_hand_landmarks
        if not left and not right:
            return None
        return np.concatenate([_hand_array(left), _hand_array(right)])

    if region == "pose":
        if not res.pose_landmarks:
            return None
        return _pose_array(res)

    if region == "legs":
        if not res.pose_landmarks:
            return None
        return _pose_array(res, LEG_POSE_IDX)

    if region == "full":
        if not (res.pose_landmarks or res.left_hand_landmarks or res.right_hand_landmarks):
            return None
        return np.concatenate([
            _pose_array(res),
            _hand_array(res.left_hand_landmarks),
            _hand_array(res.right_hand_landmarks),
        ])

    return None


def normalize_vector(vec: np.ndarray, region: str) -> np.ndarray:
    """
    Translation/scale-invariant normalisation.
    Hands: wrist-relative per hand. Pose/legs/full: hip-centre relative.
    Only x,y,z are shifted/scaled; visibility (pose) is left untouched.
    """
    if region == "hands":
        pts = vec.reshape(2, 21, 3).copy()
        for h in range(2):
            if np.any(pts[h]):
                wrist = pts[h, 0].copy()
                pts[h] -= wrist
                scale = np.linalg.norm(pts[h], axis=1).max()
                if scale > 1e-6:
                    pts[h] /= scale
        return pts.reshape(-1)

    if region in ("pose", "legs"):
        n = 33 if region == "pose" else len(LEG_POSE_IDX)
        pts = vec.reshape(n, 4).copy()
        xyz = pts[:, :3]
        centre = xyz[np.any(xyz != 0, axis=1)].mean(axis=0) if np.any(xyz) else 0
        xyz -= centre
        scale = np.linalg.norm(xyz, axis=1).max()
        if scale > 1e-6:
            xyz /= scale
        pts[:, :3] = xyz
        return pts.reshape(-1)

    if region == "full":
        pose = normalize_vector(vec[:33 * 4], "pose")
        hands = normalize_vector(vec[33 * 4:], "hands")
        return np.concatenate([pose, hands])

    return vec


# --------------------------------------------------------------------------- #
# Capture worker
# --------------------------------------------------------------------------- #
class CaptureWorker(QThread):
    frame_ready = Signal(object, object)   # (bgr_frame, res)
    finished_source = Signal()

    def __init__(self, source=0, loop_video=False, parent=None):
        super().__init__(parent)
        self.source = source
        self.loop_video = loop_video
        self._running = True

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.finished_source.emit()
            return

        is_file = not isinstance(self.source, int)
        delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0
        draw = mp.solutions.drawing_utils
        mh = mp.solutions.holistic

        while self._running:
            ok, frame = cap.read()
            if not ok:
                if is_file and self.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            if res.pose_landmarks:
                draw.draw_landmarks(frame, res.pose_landmarks, mh.POSE_CONNECTIONS)
            if res.left_hand_landmarks:
                draw.draw_landmarks(frame, res.left_hand_landmarks, mh.HAND_CONNECTIONS)
            if res.right_hand_landmarks:
                draw.draw_landmarks(frame, res.right_hand_landmarks, mh.HAND_CONNECTIONS)

            self.frame_ready.emit(frame, res)
            if is_file and delay:
                time.sleep(delay)

        cap.release()
        holistic.close()
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
        self.out_dir.mkdir(parents=True, exist_ok=True)
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
        self._save_labels()
        return True

    def _dir(self, mode: str, region: str, label: str) -> Path:
        d = self.out_dir / mode / region / label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _append_manifest(self, label, region, mode, frames, dim, path):
        new = not self._manifest_file.exists()
        with self._manifest_file.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["uid", "label", "region", "mode", "frames", "dim", "path"])
            w.writerow([path.stem, label, region, mode, frames, dim, str(path)])

    def save_static(self, label, region, vec) -> Path:
        if self.normalize:
            vec = normalize_vector(vec, region)
        uid = uuid.uuid4().hex[:12]
        path = self._dir("static", region, label) / f"{uid}.npy"
        np.save(path, vec.astype(np.float32))
        self._append_manifest(label, region, "static", 1, vec.shape[0], path)
        return path

    def save_sequence(self, label, region, seq) -> Path:
        arr = np.stack(
            [normalize_vector(v, region) if self.normalize else v for v in seq]
        ).astype(np.float32)
        uid = uuid.uuid4().hex[:12]
        path = self._dir("sequence", region, label) / f"{uid}.npy"
        np.save(path, arr)
        self._append_manifest(label, region, "sequence", arr.shape[0], arr.shape[1], path)
        return path

    def count(self, label, region, mode) -> int:
        d = self.out_dir / mode / region / label
        return len(list(d.glob("*.npy"))) if d.exists() else 0


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class GestoLabeller(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesto Labeller")
        self.resize(1120, 740)

        self.dataset: Dataset | None = None
        self.worker: CaptureWorker | None = None
        self.last_res = None
        self.recording = False
        self.seq_buffer: list[np.ndarray] = []

        self._build_ui()
        QShortcut(QKeySequence(Qt.Key_Space), self, self.on_capture)
        self.statusBar().showMessage("Pick an output folder to begin.")

    # ---- UI ---------------------------------------------------------------- #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # left: video
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
        for b in (self.btn_webcam, self.btn_video, self.btn_stop):
            src.addWidget(b)
        left.addLayout(src)
        root.addLayout(left, 2)

        # right: controls
        right = QVBoxLayout()

        # output
        out_box = QGroupBox("Output")
        ob = QVBoxLayout(out_box)
        self.btn_out = QPushButton("Choose Folder…")
        self.btn_out.clicked.connect(self.choose_output)
        self.lbl_out = QLabel("—")
        self.lbl_out.setWordWrap(True)
        self.chk_norm = QCheckBox("Normalise landmarks")
        self.chk_norm.setChecked(True)
        ob.addWidget(self.btn_out)
        ob.addWidget(self.lbl_out)
        ob.addWidget(self.chk_norm)
        right.addWidget(out_box)

        # region
        region_box = QGroupBox("Region")
        rgl = QVBoxLayout(region_box)
        self.cmb_region = QComboBox()
        for key, meta in REGIONS.items():
            self.cmb_region.addItem(f"{meta['label']}  (dim {meta['dim']})", key)
        self.cmb_region.currentIndexChanged.connect(self.refresh_labels)
        rgl.addWidget(self.cmb_region)
        right.addWidget(region_box)

        # mode
        mode_box = QGroupBox("Mode")
        mb = QHBoxLayout(mode_box)
        self.rb_static = QRadioButton("Static")
        self.rb_seq = QRadioButton("Sequence")
        self.rb_static.setChecked(True)
        self.rb_static.toggled.connect(self.refresh_labels)
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

    # ---- helpers ----------------------------------------------------------- #
    def current_region(self) -> str:
        return self.cmb_region.currentData()

    def current_mode(self) -> str:
        return "static" if self.rb_static.isChecked() else "sequence"

    def current_label(self) -> str | None:
        it = self.label_list.currentItem()
        return it.data(Qt.UserRole) if it else None

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
        name = self.in_label.text().strip()
        if self.dataset.add_label(name):
            self.in_label.clear()
            self.refresh_labels(select=name)

    def refresh_labels(self, *_, select: str | None = None):
        """Rebuild the label list while preserving the current selection."""
        if not self.dataset:
            return
        # remember what was selected before rebuilding
        keep = select or self.current_label()

        self.label_list.blockSignals(True)
        self.label_list.clear()
        region, mode = self.current_region(), self.current_mode()
        row_to_select = 0
        for i, name in enumerate(self.dataset.labels):
            n = self.dataset.count(name, region, mode)
            item = QListWidgetItem(f"{name}   ({n})")
            item.setData(Qt.UserRole, name)
            self.label_list.addItem(item)
            if name == keep:
                row_to_select = i
        self.label_list.blockSignals(False)

        if self.label_list.count():
            self.label_list.setCurrentRow(row_to_select)

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
    def on_frame(self, frame, res):
        self.last_res = res

        if self.recording:
            vec = extract_vector(res, self.current_region())
            if vec is not None:
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
        region = self.current_region()

        if self.current_mode() == "static":
            vec = extract_vector(self.last_res, region)
            if vec is None:
                self.statusBar().showMessage("Nothing detected for this region.")
                return
            p = self.dataset.save_static(label, region, vec)
            self.statusBar().showMessage(f"Saved static [{label}] → {p.name}")
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
        region = self.current_region()
        if label and len(self.seq_buffer) >= 2:
            p = self.dataset.save_sequence(label, region, self.seq_buffer)
            self.statusBar().showMessage(
                f"Saved sequence [{label}] ({len(self.seq_buffer)} frames) → {p.name}"
            )
        else:
            self.statusBar().showMessage("Sequence too short — discarded.")
        self.seq_buffer.clear()
        self.lbl_rec.setText("")
        self.refresh_labels()

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