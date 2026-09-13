"""
Builds, compares, evaluates and saves the face classification models.

Four candidate pipelines are trained and compared by cross-validation, and
the best one is kept. All four use algorithms covered in a standard machine
learning course: PCA, SVM, KNN and Random Forest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import cv2

from .features import FlattenPixels, HOGDescriptor


def balance_with_flips(X: np.ndarray, y: np.ndarray,
                       cap: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Even out class sizes by adding horizontally flipped copies.

    A face is still recognisably the same person when mirrored, so a flip is
    a free extra training example. Players with few images get flips added
    until they approach the size of the largest class, which stops the
    classifier from learning that guessing the most photographed player is a
    safe bet.

    This must only ever be called on the training split. Flipping before the
    split would put a mirrored copy of a test image into training, and the
    reported accuracy would be measuring memorisation.
    """
    labels, counts = np.unique(y, return_counts=True)
    target = int(counts.max() * cap)

    extra_X: list[np.ndarray] = []
    extra_y: list[str] = []

    for label, count in zip(labels, counts):
        shortfall = target - count
        if shortfall <= 0:
            continue
        pool = X[y == label]
        for index in range(min(shortfall, len(pool))):
            extra_X.append(cv2.flip(pool[index], 1))
            extra_y.append(label)

    if not extra_X:
        return X, y
    return np.concatenate([X, np.array(extra_X)]), np.concatenate([y, np.array(extra_y)])

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_model.joblib"


def build_pipelines(n_components: int = 60, random_state: int = 42) -> dict[str, Pipeline]:
    """
    The candidate models.

    `n_components` is capped at training time, because PCA cannot extract
    more components than there are samples or pixels.
    """
    return {
        "Eigenfaces + SVM": Pipeline([
            ("pixels", FlattenPixels()),
            ("pca", PCA(n_components=n_components, whiten=True, random_state=random_state)),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale",
                        class_weight="balanced", probability=True,
                        random_state=random_state)),
        ]),
        "HOG + SVM": Pipeline([
            ("hog", HOGDescriptor()),
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="linear", C=1.0, class_weight="balanced",
                        probability=True, random_state=random_state)),
        ]),
        "Eigenfaces + KNN": Pipeline([
            ("pixels", FlattenPixels()),
            ("pca", PCA(n_components=n_components, whiten=True, random_state=random_state)),
            ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance")),
        ]),
        "HOG + Random Forest": Pipeline([
            ("hog", HOGDescriptor()),
            ("forest", RandomForestClassifier(n_estimators=300, random_state=random_state,
                                              class_weight="balanced", n_jobs=-1)),
        ]),
    }


@dataclass
class TrainingResult:
    """Everything produced by one training run."""
    model: Pipeline
    model_name: str
    labels: list[str]
    test_accuracy: float
    cv_scores: dict[str, float] = field(default_factory=dict)
    report: str = ""
    confusion: np.ndarray | None = None
    n_train: int = 0
    n_test: int = 0
    n_components: int = 0
    n_augmented: int = 0
    failures: dict[str, str] = field(default_factory=dict)


def _safe_components(requested: int, n_samples: int, n_features: int,
                     cv_folds: int = 1) -> int:
    """
    PCA cannot ask for more components than samples or features allow.

    The binding constraint is the smallest cross-validation fold, not the
    whole training set: with 75 images and 3 folds, each fit sees only 50,
    so asking for 60 components makes every fold fail. Sizing against the
    full training set instead is a silent bug, because the failures surface
    as a NaN score and the model is quietly dropped from the comparison.
    """
    smallest_fit = n_samples
    if cv_folds > 1:
        smallest_fit = int(n_samples * (cv_folds - 1) / cv_folds)
    return max(2, min(requested, smallest_fit - 1, n_features))


def train(
    faces: list[np.ndarray],
    labels: list[str],
    test_size: float = 0.25,
    n_components: int = 60,
    cv_folds: int = 3,
    random_state: int = 42,
    compare_all: bool = True,
    balance: bool = True,
) -> TrainingResult:
    """
    Train on the given face crops and return the best model.

    Raises ValueError with a readable message when the dataset is too small
    or too lopsided to train on, which is the most common problem with a
    hand-assembled image dataset.
    """
    if len(faces) != len(labels):
        raise ValueError("Got a different number of faces and labels.")

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        raise ValueError(
            f"Need at least 2 different people to train a classifier, found {len(unique)}."
        )

    smallest = counts.min()
    if smallest < 4:
        worst = unique[counts.argmin()]
        raise ValueError(
            f"'{worst}' has only {smallest} usable face image(s). "
            f"Every person needs at least 4, and 20 or more works far better."
        )

    X = np.array(faces)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Balance AFTER the split: see balance_with_flips for why this order matters.
    n_before = len(X_train)
    if balance:
        X_train, y_train = balance_with_flips(X_train, y_train)
    n_added = len(X_train) - n_before

    # Cross-validate on the training split only, so the test split stays unseen.
    folds = max(2, min(cv_folds, np.bincount(
        np.unique(y_train, return_inverse=True)[1]).min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    components = _safe_components(n_components, len(X_train), X[0].size, folds)
    pipelines = build_pipelines(n_components=components, random_state=random_state)
    if not compare_all:
        pipelines = {"Eigenfaces + SVM": pipelines["Eigenfaces + SVM"]}

    cv_scores: dict[str, float] = {}
    failures: dict[str, str] = {}
    for name, pipeline in pipelines.items():
        try:
            scores = cross_val_score(pipeline, X_train, y_train, cv=splitter,
                                     n_jobs=1, error_score="raise")
            cv_scores[name] = float(scores.mean())
        except Exception as error:
            cv_scores[name] = float("nan")
            failures[name] = f"{type(error).__name__}: {error}"

    usable = {k: v for k, v in cv_scores.items() if not np.isnan(v)}
    if not usable:
        detail = "; ".join(f"{k} ({v})" for k, v in failures.items())
        raise ValueError(f"No model could be trained on this dataset. {detail}")

    best_name = max(usable, key=usable.get)
    best = pipelines[best_name]
    best.fit(X_train, y_train)

    predictions = best.predict(X_test)
    label_order = sorted(unique.tolist())

    return TrainingResult(
        model=best,
        model_name=best_name,
        labels=label_order,
        test_accuracy=float(accuracy_score(y_test, predictions)),
        cv_scores=cv_scores,
        report=classification_report(y_test, predictions, zero_division=0),
        confusion=confusion_matrix(y_test, predictions, labels=label_order),
        n_train=len(X_train),
        n_test=len(X_test),
        n_components=components,
        n_augmented=n_added,
        failures=failures,
    )


def predict(model: Pipeline, face: np.ndarray) -> tuple[str, float, dict[str, float]]:
    """
    Identify one face.

    Returns the predicted name, the confidence in it, and the full set of
    per-person probabilities so the interface can show the runners-up.
    A confident wrong answer is worse than an uncertain one, so the caller
    should always show the confidence alongside the name.
    """
    batch = np.array([face])
    name = str(model.predict(batch)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(batch)[0]
        classes = [str(c) for c in model.classes_]
        ranked = dict(sorted(zip(classes, probabilities.tolist()),
                             key=lambda item: item[1], reverse=True))
        return name, float(max(probabilities)), ranked

    return name, float("nan"), {}


def save_model(result: TrainingResult, path: Path = MODEL_PATH) -> Path:
    """Persist the trained model along with the metadata needed to describe it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "model_name": result.model_name,
            "labels": result.labels,
            "test_accuracy": result.test_accuracy,
            "cv_scores": result.cv_scores,
            "n_train": result.n_train,
            "n_test": result.n_test,
        },
        path,
    )
    return path


def load_model(path: Path = MODEL_PATH) -> dict | None:
    """Load a previously trained model, or None if there isn't one yet."""
    if not Path(path).exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

@dataclass
class StabilityReport:
    """Accuracy of every model across many different random splits."""
    scores: dict[str, list[float]] = field(default_factory=dict)
    n_repeats: int = 0
    n_test: int = 0

    def mean(self, name: str) -> float:
        return float(np.mean(self.scores[name]))

    def std(self, name: str) -> float:
        return float(np.std(self.scores[name]))

    def best(self) -> str:
        return max(self.scores, key=self.mean)

    def margin_of_error(self) -> float:
        """
        Roughly how far apart two results must be before the difference is
        worth believing, as a 95% interval on a single split's accuracy.

        With around 100 test images, this lands near 10 percentage points.
        Any two configurations closer than that are indistinguishable, and
        tuning against such a gap is fitting to noise.
        """
        if not self.scores or self.n_test <= 0:
            return 0.0
        p = float(np.mean([np.mean(v) for v in self.scores.values()]))
        return float(1.96 * np.sqrt(max(p * (1 - p), 1e-9) / self.n_test))


def evaluate_stability(
    faces: list[np.ndarray],
    labels: list[str],
    n_repeats: int = 10,
    test_size: float = 0.25,
    n_components: int = 60,
    balance: bool = True,
    progress=None,
) -> StabilityReport:
    """
    Train every model on many different random splits of the same data.

    A single train/test split gives one number with no indication of how much
    it would move if the split had fallen differently. Repeating the split
    turns that one number into a mean and a spread, which is what makes it
    possible to say whether one configuration genuinely beats another or
    whether the gap is just which images happened to land in the test set.
    """
    X, y = np.array(faces), np.array(labels)
    report = StabilityReport(n_repeats=n_repeats)
    report.scores = {name: [] for name in build_pipelines()}

    total = n_repeats * len(report.scores)
    step = 0

    for repeat in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=repeat, stratify=y
        )
        report.n_test = len(X_test)

        if balance:
            X_train, y_train = balance_with_flips(X_train, y_train)

        components = _safe_components(n_components, len(X_train), X[0].size)
        pipelines = build_pipelines(n_components=components, random_state=repeat)

        for name, pipeline in pipelines.items():
            try:
                pipeline.fit(X_train, y_train)
                report.scores[name].append(
                    float(accuracy_score(y_test, pipeline.predict(X_test)))
                )
            except Exception:
                pass
            step += 1
            if progress is not None:
                progress(step / total, f"split {repeat + 1} of {n_repeats}: {name}")

    report.scores = {k: v for k, v in report.scores.items() if v}
    return report
