"""
Turns a face crop into a feature vector a classifier can learn from.

Two approaches are offered, both standard in the classical face recognition
literature:

  Eigenfaces - flatten the pixels and reduce them with PCA. The principal
  components are themselves face-like images ("eigenfaces"), and a face is
  represented by how much of each component it contains.

  HOG - histogram of oriented gradients. Describes local edge directions
  rather than raw brightness, which makes it more tolerant of lighting
  changes than raw pixels.

Both are wrapped as scikit-learn transformers so they drop straight into a
Pipeline alongside any classifier.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin

from .detection import FACE_SIZE


class FlattenPixels(BaseEstimator, TransformerMixin):
    """Face images to a 2D array of raw pixel values, scaled to 0-1."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=np.float32)
        return array.reshape(len(array), -1) / 255.0


class HOGDescriptor(BaseEstimator, TransformerMixin):
    """
    Histogram of oriented gradients for each face.

    The defaults (9 orientations, 8x8 pixel cells, 2x2 cell blocks) are the
    values from the original Dalal and Triggs paper and work well on faces
    normalised to 100x100.
    """

    def __init__(self, orientations: int = 9, pixels_per_cell: int = 8,
                 cells_per_block: int = 2):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            hog(
                np.asarray(image, dtype=np.float32),
                orientations=self.orientations,
                pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                cells_per_block=(self.cells_per_block, self.cells_per_block),
                block_norm="L2-Hys",
                feature_vector=True,
            )
            for image in X
        ])


def eigenface_images(pca, shape: tuple[int, int] = FACE_SIZE) -> np.ndarray:
    """
    The principal components reshaped back into images.

    Useful for showing what the model has actually learned: the first few
    eigenfaces usually capture lighting and head shape, later ones capture
    finer facial structure.
    """
    return pca.components_.reshape((-1, *shape))
