"""
Finds faces in an image and turns them into standardised crops.

Everything downstream trains on the output of this module, so consistency
matters more than cleverness here: every face that comes out is the same
size, the same colour space, and roughly the same orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Every face is normalised to this size before feature extraction.
FACE_SIZE = (100, 100)

# Haar cascades ship with OpenCV, so there is nothing to download.
#
# OpenCV 5.0 dropped CascadeClassifier from the plain opencv-python build,
# so requirements.txt pins to the 4.x line. Failing here with a clear
# message beats an AttributeError from somewhere deep in an import.
if not hasattr(cv2, "CascadeClassifier"):
    raise ImportError(
        f"This project needs OpenCV 4.x, but version {cv2.__version__} is "
        f"installed and does not provide CascadeClassifier. Fix it with:\n"
        f'    pip install "opencv-python>=4.8,<5"'
    )

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if _FACE_CASCADE.empty() or _EYE_CASCADE.empty():
    raise ImportError(
        "OpenCV's Haar cascade files could not be loaded. Reinstall with:\n"
        '    pip install --force-reinstall "opencv-python>=4.8,<5"'
    )


@dataclass
class DetectedFace:
    """One face found in an image."""
    image: np.ndarray          # the normalised grayscale crop
    box: tuple[int, int, int, int]   # x, y, w, h in the original image
    aligned: bool              # whether eye-based rotation was applied


def read_image(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes into a BGR array. Returns None if unreadable."""
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _align_by_eyes(gray_face: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Rotate the crop so the eyes sit on a horizontal line.

    Cricket photographs are taken mid-action, so heads are often tilted.
    Straightening them before training removes a large source of variation
    that has nothing to do with who the person is.
    """
    eyes = _EYE_CASCADE.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
    if len(eyes) < 2:
        return gray_face, False

    # Use the two largest detections, which are almost always the real eyes.
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])  # left eye first
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes

    centre1 = (x1 + w1 / 2, y1 + h1 / 2)
    centre2 = (x2 + w2 / 2, y2 + h2 / 2)

    delta_y = centre2[1] - centre1[1]
    delta_x = centre2[0] - centre1[0]
    if delta_x == 0:
        return gray_face, False

    angle = np.degrees(np.arctan2(delta_y, delta_x))
    if abs(angle) > 35:      # implausible tilt: the detections were probably wrong
        return gray_face, False

    centre = ((centre1[0] + centre2[0]) / 2, (centre1[1] + centre2[1]) / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_face, matrix, (gray_face.shape[1], gray_face.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, True


def normalise_face(gray_face: np.ndarray, align: bool = True) -> tuple[np.ndarray, bool]:
    """Resize, optionally align, and equalise the histogram of one face crop."""
    aligned = False
    if align:
        gray_face, aligned = _align_by_eyes(gray_face)

    resized = cv2.resize(gray_face, FACE_SIZE, interpolation=cv2.INTER_AREA)

    # Histogram equalisation flattens out differences in stadium lighting,
    # floodlit night matches versus daytime, and so on.
    return cv2.equalizeHist(resized), aligned


def detect_faces(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbours: int = 5,
    min_size: int = 40,
    align: bool = True,
) -> list[DetectedFace]:
    """
    Find every face in a BGR image and return normalised crops.

    Raising min_neighbours reduces false positives (crowd textures picked up
    as faces) at the cost of missing some real faces.
    """
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbours,
        minSize=(min_size, min_size),
    )

    faces: list[DetectedFace] = []
    for (x, y, w, h) in boxes:
        # A small margin keeps hairline and chin, which carry identity information.
        margin = int(0.1 * w)
        top = max(y - margin, 0)
        left = max(x - margin, 0)
        bottom = min(y + h + margin, gray.shape[0])
        right = min(x + w + margin, gray.shape[1])

        crop = gray[top:bottom, left:right]
        if crop.size == 0:
            continue

        normalised, aligned = normalise_face(crop, align=align)
        faces.append(DetectedFace(image=normalised, box=(x, y, w, h), aligned=aligned))

    return faces


def largest_face(image: np.ndarray, **kwargs) -> DetectedFace | None:
    """
    The biggest face in the image, which in a portrait or a player photo is
    almost always the subject rather than someone in the background.
    """
    faces = detect_faces(image, **kwargs)
    if not faces:
        return None
    return max(faces, key=lambda face: face.box[2] * face.box[3])


def draw_boxes(image: np.ndarray, faces: list[DetectedFace],
               labels: list[str] | None = None) -> np.ndarray:
    """Return a copy of the image with detection boxes drawn on it."""
    annotated = image.copy()
    for index, face in enumerate(faces):
        x, y, w, h = face.box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (31, 111, 99), 2)
        if labels and index < len(labels):
            cv2.putText(
                annotated, labels[index], (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (31, 111, 99), 2, cv2.LINE_AA,
            )
    return annotated
