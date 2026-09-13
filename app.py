"""
Indian Cricket Team Member Face Identification
----------------------------------------------
Build a dataset from labelled player photographs, train a classifier on the
detected faces, and identify who is in a new image.

Run it with:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.dataset import build_dataset, list_people, save_crops
from src.detection import detect_faces, read_image, draw_boxes
from src.features import eigenface_images
from src.model import (MODEL_PATH, evaluate_stability, load_model, predict,
                       save_model, train)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Cricket Face Identification",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');
    :root { --ink:#1B2733; --muted:#5C6B7A; --rule:#CFD8E0; --signal:#1F6F63; --flag:#9B3B2F; }
    html, body, [class*="css"], .stMarkdown { font-family:'IBM Plex Sans', system-ui, sans-serif; color:var(--ink); }
    h1, h2, h3 { font-family:'IBM Plex Serif', Georgia, serif; letter-spacing:-0.01em; }
    h1 { font-size:2.1rem; margin-bottom:0.1rem; }
    .lede { color:var(--muted); max-width:64ch; line-height:1.55; margin-bottom:1.4rem; }
    .verdict { border:1px solid var(--rule); border-left:3px solid var(--signal);
               padding:0.9rem 1.1rem; margin-bottom:0.6rem; }
    .verdict-name { font-family:'IBM Plex Serif', Georgia, serif; font-size:1.5rem; font-weight:600; }
    .verdict-low { border-left-color:var(--flag); }
    .muted { color:var(--muted); font-size:0.86rem; }
    .empty { border:1px dashed var(--rule); padding:2rem; text-align:center; color:var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cricket face identification")
st.markdown(
    "<p class='lede'>Train a classifier on photographs of Indian cricket team "
    "members, then identify who appears in a new image. Faces are detected with "
    "a Haar cascade, straightened using eye positions, and classified with "
    "models built from PCA, SVM, KNN and Random Forest.</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    dataset_path = st.text_input("Dataset folder", value=str(RAW_DIR))
    align_faces = st.checkbox("Straighten faces using eye positions", value=True)
    min_neighbours = st.slider(
        "Detection strictness", 3, 12, 5,
        help="Higher values reject more false detections but miss more real faces. "
             "Lowering it adds faces, but the extra ones are often junk.",
    )
    min_sharpness = st.slider(
        "Minimum crop quality", 0, 400, 100, 25,
        help="Rejects blurred or featureless crops using the variance of the "
             "Laplacian. Set to 0 to keep every detection.",
    )
    st.divider()
    st.caption("Training")
    test_size = st.slider("Held-out test share", 0.1, 0.4, 0.25, 0.05)
    n_components = st.slider("PCA components (eigenfaces)", 10, 150, 60, 10)
    compare_all = st.checkbox("Compare all four models", value=True)
    balance = st.checkbox(
        "Balance classes with flipped copies", value=True,
        help="Adds horizontally flipped images to under-represented players. "
             "Applied only to the training split, never the test split.",
    )

    st.divider()
    saved = load_model()
    if saved:
        st.success(f"Trained model ready\n\n{saved['model_name']}")
        st.caption(f"Test accuracy {saved['test_accuracy']:.1%} · "
                   f"{len(saved['labels'])} people")
    else:
        st.info("No trained model yet. Use the Train tab.")

tab_data, tab_train, tab_identify = st.tabs(["1. Dataset", "2. Train", "3. Identify"])

# --------------------------------------------------------------------------
# Tab 1: dataset
# --------------------------------------------------------------------------

with tab_data:
    st.subheader("Dataset")
    st.markdown(
        "Put one folder per player inside the dataset folder, with that "
        "player's photographs inside it. The folder name becomes the label "
        "the model predicts."
    )
    st.code(
        "data/raw/\n"
        "├── Virat Kohli/\n"
        "│   ├── 001.jpg\n"
        "│   └── 002.jpg\n"
        "└── Rohit Sharma/\n"
        "    ├── 001.jpg\n"
        "    └── 002.jpg",
        language="text",
    )

    people = list_people(Path(dataset_path))
    if not people:
        st.markdown(
            "<div class='empty'>No player folders found yet.<br>"
            "Add them to the dataset folder shown in the sidebar, then reload.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.write(f"Found **{len(people)}** people: {', '.join(people)}")

        if st.button("Scan images and detect faces", width="stretch"):
            bar = st.progress(0.0, text="Starting")
            report = build_dataset(
                Path(dataset_path), align=align_faces,
                min_neighbours=min_neighbours,
                min_sharpness=float(min_sharpness),
                progress=lambda fraction, text: bar.progress(fraction, text=text),
            )
            bar.empty()
            st.session_state["report"] = report

        report = st.session_state.get("report")
        if report:
            st.success(report.summary())

            if report.per_person:
                counts = pd.DataFrame(
                    sorted(report.per_person.items()), columns=["Person", "Usable faces"]
                )
                st.bar_chart(counts.set_index("Person"), horizontal=True)

                thin = [p for p, c in report.per_person.items() if c < 20]
                if thin:
                    st.warning(
                        "Fewer than 20 usable faces for: " + ", ".join(sorted(thin)) +
                        ". The model will be unreliable for these people. Add more "
                        "photographs, or lower the detection strictness."
                    )

            if report.rejected_crops:
                with st.expander(
                    f"Rejected {len(report.skipped_low_quality)} crops as blurred or featureless"
                ):
                    st.caption(
                        "These passed the face detector but contain too little detail "
                        "to be a usable face. If real faces are being rejected here, "
                        "lower the minimum crop quality."
                    )
                    sample = report.rejected_crops[:12]
                    columns = st.columns(min(6, len(sample)))
                    for position, crop in enumerate(sample):
                        with columns[position % len(columns)]:
                            st.image(crop, width="stretch", clamp=True)
                    st.text("\n".join(report.skipped_low_quality[:40]))

            if report.skipped_no_face:
                with st.expander(f"No face detected in {len(report.skipped_no_face)} images"):
                    st.caption(
                        "Usually side-on action shots, helmets, sunglasses, or faces "
                        "too small in the frame. Lowering detection strictness recovers some."
                    )
                    st.text("\n".join(report.skipped_no_face[:60]))

            if report.faces:
                st.markdown("**Sample of the detected faces**")
                sample_index = np.linspace(0, len(report.faces) - 1,
                                           min(12, len(report.faces))).astype(int)
                columns = st.columns(min(6, len(sample_index)))
                for position, index in enumerate(sample_index):
                    with columns[position % len(columns)]:
                        st.image(report.faces[index], caption=report.labels[index],
                                 width="stretch", clamp=True)

                if st.button("Save these crops to data/processed"):
                    written = save_crops(report, PROCESSED_DIR)
                    st.success(f"Wrote {written} face crops to {PROCESSED_DIR}")

# --------------------------------------------------------------------------
# Tab 2: training
# --------------------------------------------------------------------------

with tab_train:
    st.subheader("Train")
    report = st.session_state.get("report")

    if not report or not report.faces:
        st.markdown(
            "<div class='empty'>Scan the dataset on the Dataset tab first.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.write(f"Ready to train on **{len(report.faces)}** faces "
                 f"from **{len(report.per_person)}** people.")

        if st.button("Train model", type="primary", width="stretch"):
            try:
                with st.spinner("Training and comparing models"):
                    result = train(
                        report.faces, report.labels,
                        test_size=test_size, n_components=n_components,
                        compare_all=compare_all, balance=balance,
                    )
                save_model(result)
                st.session_state["result"] = result
            except ValueError as error:
                st.error(str(error))

        result = st.session_state.get("result")
        if result:
            st.success(f"Trained: {result.model_name}")

            columns = st.columns(3)
            columns[0].metric("Test accuracy", f"{result.test_accuracy:.1%}")
            columns[1].metric(
                "Training images", result.n_train,
                help=f"{result.n_augmented} are flipped copies added to balance classes"
                     if result.n_augmented else None,
            )
            columns[2].metric("Test images", result.n_test)

            if result.n_augmented:
                st.caption(
                    f"{result.n_augmented} flipped copies were added to the training "
                    f"split to even out class sizes. The test split is untouched."
                )

            if len(result.cv_scores) > 1:
                st.markdown("**Model comparison (cross-validation on training data)**")
                comparison = pd.DataFrame(
                    sorted(result.cv_scores.items(), key=lambda item: -item[1]),
                    columns=["Model", "CV accuracy"],
                )
                st.dataframe(comparison, hide_index=True, width="stretch")

            if result.n_components and result.n_components < n_components:
                st.info(
                    f"PCA used {result.n_components} components instead of the "
                    f"{n_components} requested, because the dataset is too small to "
                    f"support more. Add images per person to use more components."
                )

            if result.failures:
                with st.expander(f"{len(result.failures)} model(s) could not be trained"):
                    for name, reason in result.failures.items():
                        st.markdown(f"- **{name}** — {reason}")

            st.markdown("**Per-person performance on the held-out test set**")
            st.code(result.report, language="text")

            if result.confusion is not None:
                st.markdown("**Confusion matrix**")
                st.caption("Rows are the true person, columns are the prediction. "
                           "Off-diagonal numbers are the mistakes.")
                matrix = pd.DataFrame(result.confusion,
                                      index=result.labels, columns=result.labels)
                st.dataframe(matrix, width="stretch")

            st.divider()
            st.markdown("**How much of this is real?**")
            st.caption(
                "A single train/test split gives one number with no sense of how far "
                "it would move if the split had fallen differently. Repeating it many "
                "times turns that into a mean and a spread, which is the only way to "
                "tell a genuine improvement from a lucky split."
            )
            repeats = st.slider("Number of random splits", 3, 20, 10)

            if st.button(f"Run {repeats} splits and measure the spread", width="stretch"):
                bar = st.progress(0.0, text="Starting")
                stability = evaluate_stability(
                    report.faces, report.labels, n_repeats=repeats,
                    test_size=test_size, n_components=n_components, balance=balance,
                    progress=lambda fraction, text: bar.progress(fraction, text=text),
                )
                bar.empty()
                st.session_state["stability"] = stability

            stability = st.session_state.get("stability")
            if stability:
                rows = []
                for name in sorted(stability.scores, key=stability.mean, reverse=True):
                    values = stability.scores[name]
                    rows.append({
                        "Model": name,
                        "Mean accuracy": f"{stability.mean(name):.1%}",
                        "Std dev": f"{stability.std(name):.1%}",
                        "Worst split": f"{min(values):.1%}",
                        "Best split": f"{max(values):.1%}",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

                margin = stability.margin_of_error()
                st.info(
                    f"With {stability.n_test} test images, a single split's accuracy "
                    f"carries a 95% margin of error of roughly ±{margin:.1%}. Two "
                    f"configurations closer together than that are not "
                    f"distinguishable — tuning against a smaller gap is fitting to "
                    f"noise rather than improving the model."
                )

                best = stability.best()
                others = [n for n in stability.scores if n != best]
                clear = [
                    n for n in others
                    if stability.mean(best) - stability.mean(n) > margin
                ]
                if clear:
                    st.success(
                        f"{best} beats {', '.join(clear)} by more than the margin of "
                        f"error, so that difference is worth believing."
                    )
                else:
                    st.warning(
                        "No model beats the others by more than the margin of error. "
                        "On this dataset the choice of model is not what is limiting "
                        "accuracy — the data is."
                    )

            st.divider()
            pca = result.model.named_steps.get("pca")
            if pca is not None:
                with st.expander("What the model learned: eigenfaces"):
                    st.caption(
                        "The principal components, drawn as images. Early ones "
                        "capture lighting and head shape; later ones capture finer "
                        "facial structure."
                    )
                    images = eigenface_images(pca)
                    columns = st.columns(6)
                    for index in range(min(12, len(images))):
                        face = images[index]
                        normalised = ((face - face.min()) /
                                      (np.ptp(face) + 1e-9) * 255).astype(np.uint8)
                        with columns[index % 6]:
                            st.image(normalised, caption=f"PC {index + 1}",
                                     width="stretch", clamp=True)

# --------------------------------------------------------------------------
# Tab 3: identification
# --------------------------------------------------------------------------

with tab_identify:
    st.subheader("Identify")
    saved = load_model()

    if not saved:
        st.markdown(
            "<div class='empty'>Train a model first, then come back here.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Using {saved['model_name']} · trained on "
                   f"{', '.join(saved['labels'])}")

        uploaded = st.file_uploader(
            "Upload a photograph",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        if uploaded is not None:
            image = read_image(uploaded.getvalue())

            if image is None:
                st.error("That file could not be read as an image.")
            else:
                faces = detect_faces(image, align=align_faces,
                                     min_neighbours=min_neighbours)

                if not faces:
                    st.warning(
                        "No face detected. Try a clearer, more front-facing "
                        "photograph, or lower the detection strictness in the sidebar."
                    )
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=400)
                else:
                    names = []
                    results = []
                    for face in faces:
                        name, confidence, ranked = predict(saved["model"], face.image)
                        names.append(f"{name} {confidence:.0%}" if confidence == confidence else name)
                        results.append((face, name, confidence, ranked))

                    annotated = draw_boxes(image, faces, names)
                    left, right = st.columns([3, 2])
                    left.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               caption=f"{len(faces)} face(s) detected", width="stretch")

                    with right:
                        for face, name, confidence, ranked in results:
                            low = confidence == confidence and confidence < 0.5
                            st.markdown(
                                f"<div class='verdict {'verdict-low' if low else ''}'>"
                                f"<div class='verdict-name'>{name}</div>"
                                f"<div class='muted'>confidence {confidence:.1%}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.image(face.image, width=110, clamp=True)

                            if low:
                                st.caption(
                                    "Low confidence. The model always returns its "
                                    "closest match even when the person is not one "
                                    "it was trained on."
                                )
                            if ranked:
                                others = list(ranked.items())[1:4]
                                if others:
                                    st.caption("Next closest: " + " · ".join(
                                        f"{n} {p:.0%}" for n, p in others))

        st.caption(
            "The classifier can only return one of the people it was trained on. "
            "Show it someone else and it will still name its closest match, which "
            "is why the confidence figure matters."
        )
