# Indian Cricket Team Member Face Identification

A Streamlit app that learns to recognise Indian cricket team members from
photographs. It detects a face in an image, straightens it, and classifies it
as one of the players it was trained on.

Built for the Edufyi Tech Solutions project submission (Problem statement 2:
Indian Cricket Team Member Face Identification).

---

## What it does

1. Reads a folder of labelled player photographs, one folder per player.
2. Detects the face in each photograph with a Haar cascade classifier.
3. Straightens each face using the positions of the eyes, then normalises it
   to 100x100 grayscale with histogram equalisation.
4. Trains four different classifiers on the detected faces and picks the best
   one by cross-validation.
5. Reports accuracy, a per-player breakdown, and a confusion matrix.
6. Identifies the player in a new uploaded photograph, with a confidence score.

---

## Method

The pipeline is built from classical computer vision and machine learning
rather than deep learning. It trains in seconds on a laptop CPU, needs no GPU
and no model downloads, and every stage can be inspected.

**Face detection.** Haar cascade classifiers, which ship with OpenCV. The
detector finds candidate face regions, and a second cascade finds the eyes.
The angle between the eyes is used to rotate the crop upright, which removes a
large source of variation in action photographs where heads are tilted.

**Normalisation.** Every face becomes a 100x100 grayscale image with its
histogram equalised, so a floodlit night match and a daytime match produce
comparable inputs.

**Features.** Two approaches are compared:

- *Eigenfaces* — the face is flattened to a vector of pixels and reduced with
  PCA. The principal components are themselves face-like images, and each face
  is described by how much of each component it contains. The app can display
  these eigenfaces so you can see what the model learned.
- *HOG* — histogram of oriented gradients, which describes local edge
  directions rather than brightness and is more tolerant of lighting changes.

**Classifiers.** Four pipelines are trained and compared by cross-validation:

| Pipeline | Features | Classifier |
|---|---|---|
| Eigenfaces + SVM | PCA | Support Vector Machine, RBF kernel |
| HOG + SVM | HOG | Support Vector Machine, linear kernel |
| Eigenfaces + KNN | PCA | K Nearest Neighbours, k=3 |
| HOG + Random Forest | HOG | Random Forest, 300 trees |

The winner is chosen on cross-validation accuracy over the training split, and
only then evaluated on a held-out test split that no model has seen. PCA
component count is sized against the smallest cross-validation fold, so the
eigenface models still train on small datasets instead of failing silently.

---

## Getting the dataset

The dataset is not included — image datasets of real people cannot be
redistributed here, and the images are large. You need to assemble it.

**Option 1 — Kaggle.** Search Kaggle for "Indian cricket players face dataset"
or "Indian cricketers images". Download, unzip, and arrange the folders as
shown below.

**Option 2 — build your own.** Collect 25 to 40 clear, front-facing
photographs of each player. More is better, and variety in lighting, angle and
age matters more than sheer count.

Arrange them like this, with one folder per player:

```
data/raw/
├── Virat Kohli/
│   ├── 001.jpg
│   └── 002.jpg
├── Rohit Sharma/
│   ├── 001.jpg
│   └── 002.jpg
└── Jasprit Bumrah/
    ├── 001.jpg
    └── 002.jpg
```

The folder name is the label the model predicts, so spell it the way you want
it displayed.

**How many images?** The app will refuse to train with fewer than 4 usable
faces per player and will warn below 20. Aim for 25 to 40. Expect to lose some
photographs to failed detection — helmets, sunglasses and side-on action shots
often defeat the detector.

---

## Setup

**Step 1 — Terminal command.** Clone the repository and enter the folder:

```bash
git clone https://github.com/YOUR-USERNAME/cricket-face-identification.git
cd cricket-face-identification
```

**Step 2 — Terminal command.** Create a virtual environment:

```bash
python -m venv venv
```

**Step 3 — Terminal command.** Activate it.

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

**Step 4 — Terminal command.** Install the dependencies:

```bash
pip install -r requirements.txt
```

**Step 5 — Terminal command.** Verify everything works before adding any
images. This runs the whole pipeline on generated data and needs no dataset:

```bash
python test_pipeline.py
```

**Step 6 — Terminal command.** Start the app:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

---

## Using it

**Dataset tab.** Point it at your dataset folder and click *Scan images and
detect faces*. It reports how many usable faces it found per player and lists
the images where no face was detected. Look at the sample crops it shows — if
a crop contains someone in the crowd rather than the player, that image is
poisoning your training set and should be removed.

**Train tab.** Click *Train model*. You get the winning model, its accuracy on
the held-out test set, the cross-validation comparison of all four models, a
per-player breakdown, and a confusion matrix showing exactly which players get
confused with which.

Below that, *Run N splits and measure the spread* retrains every model on many
different random splits and reports a mean and standard deviation instead of a
single number. Use it before concluding that any change helped. With around
100 test images, a single split's accuracy carries a 95% margin of error near
±10 percentage points, so two configurations that differ by less than that are
indistinguishable. The app states its own margin of error and says plainly
whether the best model's lead is larger than it.

**Identify tab.** Upload a photograph. The app draws a box around each face it
finds and names the closest match with a confidence score.

---

## Project structure

```
cricket-face-identification/
├── app.py                  Streamlit interface
├── test_pipeline.py        Self-contained check, needs no dataset
├── requirements.txt        Dependencies
├── README.md               This file
├── .streamlit/
│   └── config.toml         Theme
├── src/
│   ├── detection.py        Haar detection, eye alignment, normalisation
│   ├── dataset.py          Folder walking and face extraction
│   ├── features.py         Eigenfaces (PCA) and HOG transformers
│   └── model.py            Training, comparison, evaluation, persistence
├── data/
│   ├── raw/                Your player folders go here
│   └── processed/          Saved face crops (optional)
└── models/
    └── face_model.joblib   Written when you train
```

---

## Known limits

**It cannot say "I don't know".** This is a closed-set classifier: it always
returns one of the players it was trained on. Show it a photograph of someone
else and it will still name its closest match, sometimes with high confidence.
In testing, random noise was classified as a known player at 85% confidence.
This is why the confidence figure is shown on every prediction, and why a
production system would need a distance threshold to reject unknown faces.

**Haar cascades miss a lot of cricket photography.** The detector wants
front-facing, reasonably lit faces. Helmets, sunglasses, side-on shots and
faces small in the frame are frequently missed. The app reports every skipped
image so the loss is visible rather than silent.

**Accuracy depends almost entirely on the dataset.** With 25 or more varied
images per player, this approach typically reaches useful accuracy on a small
number of players. With 5 images each, or with photographs that all come from
the same match, it will overfit and the reported accuracy will flatter it.

**The test accuracy is optimistic if images are near-duplicates.** If several
photographs come from the same burst or the same match, near-identical images
can land in both the training and test splits, which inflates the score. Vary
the sources.

**A single accuracy figure is not precise enough to tune against.** On a test
set of about 100 images, the 95% margin of error is roughly ±10 points. In
testing, three different configurations scored 62.5%, 57.8% and 56.7% — a
spread entirely inside that margin, and therefore not evidence that any of
them is better. One player's f1 score moved from 0.75 to 0.00 between runs on
5 test images, which is the sample size talking, not the model. Use the
repeated-splits evaluation before believing any improvement.

**Grayscale only.** Skin tone and kit colour are discarded, which loses some
information but avoids the model learning to identify players by their jersey
rather than their face.

---

## Possible extensions

- Replace the Haar cascade with a DNN face detector, or MTCNN, for far better
  detection on difficult photographs.
- Replace the PCA and HOG features with embeddings from a pretrained face
  recognition network such as FaceNet or ArcFace. This is the single biggest
  accuracy improvement available, at the cost of a large dependency.
- Add a confidence threshold below which the app reports "unknown" rather than
  guessing.
- Add data augmentation (flips, small rotations, brightness changes) to get
  more out of a small dataset. Horizontal flips are already implemented as an
  option; they are applied only to the training split, because flipping before
  the split would place a mirrored copy of a test image into training and make
  the reported accuracy meaningless.
- Collect more images per player. With class sizes between 13 and 45 and a
  quarter of images lost to failed detection, the dataset — not the choice of
  classifier — is what limits accuracy here.
