"""
Turns a folder of player photographs into a training set.

Expected layout - one folder per person, named after them:

    data/raw/
    ├── Virat Kohli/
    │   ├── 001.jpg
    │   └── 002.jpg
    └── Rohit Sharma/
        ├── 001.jpg
        └── 002.jpg

The folder name is the label, so whatever you call the folder is what the
model will predict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import cv2 as _cv2

from .detection import largest_face


def sharpness(face) -> float:
    """
    How much fine detail a crop contains, as the variance of its
    Laplacian. Blurred crops and flat regions such as a shirt or a
    patch of turf score near zero; a real face in focus scores high.

    Haar cascades produce a steady trickle of these false positives, and
    they are worse than useless: the model learns that a grey blur is
    whichever player it was filed under.
    """
    return float(_cv2.Laplacian(face, _cv2.CV_64F).var())

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetReport:
    """What happened while building the dataset."""
    faces: list[np.ndarray] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    per_person: dict[str, int] = field(default_factory=dict)
    skipped_no_face: list[str] = field(default_factory=list)
    skipped_unreadable: list[str] = field(default_factory=list)
    skipped_low_quality: list[str] = field(default_factory=list)
    rejected_crops: list[np.ndarray] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return (len(self.faces) + len(self.skipped_no_face)
                + len(self.skipped_unreadable) + len(self.skipped_low_quality))

    @property
    def people(self) -> list[str]:
        return sorted(self.per_person)

    def summary(self) -> str:
        parts = [
            f"{len(self.faces)} faces from {len(self.per_person)} people.",
            f"Skipped {len(self.skipped_no_face)} images with no detectable face",
        ]
        if self.skipped_low_quality:
            parts.append(f", {len(self.skipped_low_quality)} blurred or featureless crops")
        parts.append(f" and {len(self.skipped_unreadable)} unreadable files.")
        return " ".join(parts[:2]) + "".join(parts[2:])


def list_people(root: Path) -> list[str]:
    """The person folders inside the dataset root."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def build_dataset(root: Path, align: bool = True,
                  min_neighbours: int = 5,
                  min_sharpness: float = 0.0,
                  progress=None) -> DatasetReport:
    """
    Walk the dataset folder, detect one face per image, and collect the crops.

    Images where no face is found are reported rather than silently dropped,
    because a person losing most of their images to failed detection is the
    single most common reason a face model trains badly.
    """
    root = Path(root)
    report = DatasetReport()

    people = list_people(root)
    image_paths: list[tuple[str, Path]] = []
    for person in people:
        for path in sorted((root / person).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append((person, path))

    for index, (person, path) in enumerate(image_paths):
        try:
            # imdecode handles unicode paths that cv2.imread mishandles on Windows.
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            image = None

        if image is None:
            report.skipped_unreadable.append(str(path.name))
        else:
            face = largest_face(image, align=align, min_neighbours=min_neighbours)
            if face is None:
                report.skipped_no_face.append(f"{person}/{path.name}")
            elif min_sharpness > 0 and sharpness(face.image) < min_sharpness:
                report.skipped_low_quality.append(f"{person}/{path.name}")
                report.rejected_crops.append(face.image)
            else:
                report.faces.append(face.image)
                report.labels.append(person)
                report.per_person[person] = report.per_person.get(person, 0) + 1

        if progress is not None:
            progress((index + 1) / len(image_paths), f"{index + 1} of {len(image_paths)} images")

    return report


def save_crops(report: DatasetReport, destination: Path) -> int:
    """
    Write the detected crops to disk as images.

    Not required for training, but it lets you look through what the detector
    actually fed the model, which is the fastest way to spot a mislabelled
    folder or a crop that caught a face in the crowd.
    """
    destination = Path(destination)
    written = 0
    for index, (face, label) in enumerate(zip(report.faces, report.labels)):
        folder = destination / label
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{index:04d}.png"), face)
        written += 1
    return written
