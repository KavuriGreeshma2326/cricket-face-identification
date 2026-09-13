"""
Checks the whole pipeline without needing the image dataset.

Two parts:

  1. Face detection is run on a real photograph that ships with scikit-image,
     which proves the Haar cascade and eye alignment work.
  2. Training, model comparison, saving, loading and prediction are run on
     generated face-like images, which proves the learning pipeline works.

Part 2 uses synthetic patterns, so the accuracy it reports says nothing about
how well the system identifies real cricketers. It only shows the machinery
is sound. Real accuracy comes from training on the real dataset.

Run it with:  python test_pipeline.py
"""

import cv2
import numpy as np

from src.detection import detect_faces, largest_face
from src.model import load_model, predict, save_model, train


def test_detection() -> bool:
    """Detect a face in a real photograph, including a rotated copy."""
    print("1. Face detection on a real photograph")
    try:
        from skimage import data
    except ImportError:
        print("   skipped: scikit-image not installed")
        return True

    bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)

    faces = detect_faces(bgr)
    print(f"   faces found: {len(faces)}")
    if not faces:
        print("   FAILED: no face detected in a clear portrait")
        return False

    face = faces[0]
    print(f"   crop size: {face.image.shape}, aligned by eyes: {face.aligned}")

    matrix = cv2.getRotationMatrix2D((256, 256), 20, 1.0)
    rotated = cv2.warpAffine(bgr, matrix, (512, 512))
    if largest_face(rotated) is None:
        print("   FAILED: lost the face when the image was rotated 20 degrees")
        return False

    print("   rotated copy: still detected")
    return True


def make_synthetic_person(seed: int, count: int = 25) -> list[np.ndarray]:
    """Generate face-like images with a structure unique to this identity."""
    rng = np.random.default_rng(seed)
    base = np.zeros((100, 100), np.uint8)
    cv2.ellipse(base, (50, 50), (32, 42), 0, 0, 360, 200, -1)
    eye_x, eye_y = 18 + seed % 6, 38 + seed % 7
    cv2.circle(base, (50 - eye_x, eye_y), 4 + seed % 3, 40, -1)
    cv2.circle(base, (50 + eye_x, eye_y), 4 + seed % 3, 40, -1)
    cv2.line(base, (50, eye_y + 6), (50, eye_y + 18 + seed % 8), 90, 2 + seed % 2)
    cv2.ellipse(base, (50, 72 + seed % 6), (12 + seed % 7, 5), 0, 0, 180, 60, 2)

    images = []
    for _ in range(count):
        image = base.copy()
        matrix = cv2.getRotationMatrix2D((50, 50), rng.uniform(-8, 8), rng.uniform(0.95, 1.05))
        image = cv2.warpAffine(image, matrix, (100, 100), borderMode=cv2.BORDER_REPLICATE)
        image = np.clip(image.astype(np.int16) + rng.integers(-20, 20), 0, 255).astype(np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.1, 1.0))
        images.append(cv2.equalizeHist(image))
    return images


def test_training() -> bool:
    """Train, compare, save, load and predict."""
    print("\n2. Training pipeline on generated data")
    faces, labels = [], []
    for index, name in enumerate(["Player A", "Player B", "Player C", "Player D"]):
        images = make_synthetic_person(index * 7 + 1)
        faces.extend(images)
        labels.extend([name] * len(images))
    print(f"   dataset: {len(faces)} images, {len(set(labels))} identities")

    result = train(faces, labels)
    print(f"   best model: {result.model_name}")
    print(f"   test accuracy: {result.test_accuracy:.1%} "
          f"(train {result.n_train}, test {result.n_test})")
    print("   cross-validation:")
    for name, score in sorted(result.cv_scores.items(), key=lambda item: -item[1]):
        print(f"     {name:<24} {score:.1%}")

    save_model(result)
    loaded = load_model()
    if loaded is None:
        print("   FAILED: model did not survive a save and load")
        return False

    name, confidence, _ = predict(loaded["model"], faces[0])
    print(f"   prediction on a known face: {name} at {confidence:.1%}")
    if name != labels[0]:
        print(f"   FAILED: expected {labels[0]}")
        return False
    return True


def test_guards() -> bool:
    """The dataset problems that should produce a readable error, not a crash."""
    print("\n3. Error handling")
    images = [np.random.randint(0, 255, (100, 100), dtype=np.uint8) for _ in range(30)]
    cases = [
        ("only one person", lambda: train(images, ["Kohli"] * 30)),
        ("a person with 2 images", lambda: train(images[:12], ["A"] * 10 + ["B"] * 2)),
        ("mismatched faces and labels", lambda: train(images[:10], ["A"] * 5)),
        ("empty dataset", lambda: train([], [])),
    ]
    passed = True
    for description, call in cases:
        try:
            call()
            print(f"   FAILED: '{description}' was not caught")
            passed = False
        except ValueError:
            print(f"   handled: {description}")
        except Exception as error:
            print(f"   FAILED: '{description}' raised {type(error).__name__}")
            passed = False
    return passed


if __name__ == "__main__":
    results = [test_detection(), test_training(), test_guards()]
    print()
    if all(results):
        print("All checks passed.")
        print("Note: the accuracy above comes from generated images and does not "
              "predict real-world accuracy on cricketer photographs.")
    else:
        print("Some checks failed. See above.")
        raise SystemExit(1)
