"""Streamlit app for collecting real driver photos/video as labelled training data.

This is a local data-collection tool, not a live webcam demo and not for showing a
supervisor - see `src/realtime.py` for that. Every saved image goes through the exact
same face detection and preprocessing as training (`src/preprocess.py`) and the live app
(`src/realtime.py`'s `load_model_for_inference`/`predict_face`, imported here rather than
duplicated), so a face crop saved here is treated identically once it is used for training.

Run:
    streamlit run app_collect.py

Nothing is written to disk on any tab until the sidebar consent checkbox is ticked.
"""
import csv
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import CLASS_NAMES, FER_DIR, MODELS_DIR, REALTIME, RESULTS_DIR, ROOT, SEED
from src.realtime import build_face_detector, load_model_for_inference, predict_face
from src.utils import ensure_dirs, set_seed

COLLECTED_DIR = ROOT / "data" / "collected"
LOG_PATH = RESULTS_DIR / "collected_log.csv"
LOG_FIELDS = [
    "timestamp", "source", "original_filename", "model_used",
    "predicted_emotion", "confidence", "corrected_label", "saved_path",
]
KEEP_PREDICTION = "(use model prediction)"
VIDEO_SUFFIX_ERROR = "Could not read this video - check it isn't corrupted and try a common format (MP4/MOV/AVI)."


# --------------------------------------------------------------------------- non-UI helpers
# Everything below is plain Python with no `st.*` calls, so tests/test_collect.py can import
# and exercise it directly without a running Streamlit session.

def list_available_models(models_dir: Path = MODELS_DIR) -> List[str]:
    """Return the stem (e.g. 'cnn_v1') of every *.keras file in `models_dir`, sorted."""
    return sorted(p.stem for p in Path(models_dir).glob("*.keras"))


def infer_model_family(model) -> str:
    """Guess which preprocessing family a loaded model needs, from its input channel count.

    1 channel -> "custom_cnn" (grayscale, scaled to [0, 1]); 3 channels -> "vgg16" (RGB,
    Keras' VGG16 preprocess_input). This reads the model itself rather than assuming a
    filename convention, so a model added later (e.g. an EfficientNet variant) is still
    listed by `list_available_models` but raises a clear error here instead of silently
    guessing the wrong preprocessing, until it gets its own case in src/preprocess.py.
    """
    channels = model.input_shape[-1]
    if channels == 1:
        return "custom_cnn"
    if channels == 3:
        return "vgg16"
    raise ValueError(
        f"No preprocessing is defined for a model with {channels} input channels. "
        "Add a case to src/preprocess.py's preprocess() (and here) for this model family first."
    )


def save_crop(crop_bgr: np.ndarray, label: str, filename_stem: str, dest_root: Path = COLLECTED_DIR) -> Path:
    """Write `crop_bgr` to `dest_root/label/<timestamp>_<filename_stem>.jpg`; returns the path."""
    label_dir = Path(dest_root) / label
    label_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename_stem)
    path = label_dir / f"{stamp}_{safe_stem}.jpg"
    cv2.imwrite(str(path), crop_bgr)
    return path


def append_log_row(row: Dict, log_path: Path = LOG_PATH) -> None:
    """Append one row to the collected-data CSV log, writing the header first if it's new."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def sample_video_frames(video_path: str, every_n: int) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_index, frame_bgr) for every `every_n`-th frame of the video at `video_path`."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % every_n == 0:
                yield index, frame
            index += 1
    finally:
        cap.release()


def fer2013_train_class_counts(fer_dir: Path = FER_DIR) -> Optional[Dict[str, int]]:
    """Count real FER2013 training images per class, for the analytics comparison chart.

    Returns None (rather than raising) if the raw dataset isn't present locally - this app
    only needs the trained models to run day to day, so a missing dataset is expected, not
    an error.
    """
    train_dir = Path(fer_dir) / "train"
    if not train_dir.exists():
        return None
    return {cls: len(list((train_dir / cls).glob("*"))) for cls in CLASS_NAMES if (train_dir / cls).exists()}


def _save_and_log(
    crop_bgr: np.ndarray, label: str, filename_stem: str, *, source: str, original_filename: str,
    model_used: str, predicted_emotion: str, confidence: float, corrected_label: str,
) -> Path:
    """Save one face crop and append its row to the log, in one call (used by the UI tabs)."""
    path = save_crop(crop_bgr, label, filename_stem)
    try:
        saved_path = str(path.relative_to(ROOT))
    except ValueError:
        saved_path = str(path)
    append_log_row({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "original_filename": original_filename,
        "model_used": model_used,
        "predicted_emotion": predicted_emotion,
        "confidence": confidence,
        "corrected_label": corrected_label,
        "saved_path": saved_path,
    })
    return path


def _decode_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
    """Decode a Streamlit UploadedFile into a BGR image array; None if it isn't a valid image."""
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- Streamlit UI

@st.cache_resource(show_spinner="Loading model...")
def _cached_model(model_path: str):
    return load_model_for_inference(model_path)


@st.cache_resource(show_spinner="Setting up face detector...")
def _cached_detector():
    return build_face_detector(REALTIME["min_face_confidence"])


def _render_image_tab(model, model_family: str, img_size: int, detect_fn, model_name: str, consent: bool) -> None:
    st.subheader("Upload images")
    files = st.file_uploader("Images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if not files:
        st.info("Upload one or more images to run inference.")
        return

    pending = []  # (crop, label, filename, predicted, confidence, corrected_label) for "Save all"
    for i, file in enumerate(files):
        st.markdown(f"**{file.name}**")
        image_bgr = _decode_uploaded_image(file)
        if image_bgr is None:
            st.warning("Could not read this file as an image - skipped.")
            continue

        result = predict_face(image_bgr, detect_fn, model, model_family, img_size, REALTIME["face_padding"])
        if result is None:
            st.warning("No face detected in this image - skipped.")
            continue

        x, y, w, h = result["box"]
        probs = result["probs"]
        top_idx = int(np.argmax(probs))
        predicted, confidence = CLASS_NAMES[top_idx], float(probs[top_idx])

        annotated = image_bgr.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        col_img, col_chart = st.columns(2)
        with col_img:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"{predicted} ({confidence * 100:.0f}%)")
        with col_chart:
            st.bar_chart(pd.Series(probs, index=CLASS_NAMES, name="probability"))

        correction = st.selectbox("Correct label if wrong", [KEEP_PREDICTION] + CLASS_NAMES, key=f"correct_{i}")
        corrected_label = "" if correction == KEEP_PREDICTION else correction
        label = corrected_label or predicted

        if st.button(f"Save '{file.name}' to dataset", key=f"save_{i}", disabled=not consent):
            _save_and_log(
                result["crop"], label, file.name, source="image_upload", original_filename=file.name,
                model_used=model_name, predicted_emotion=predicted, confidence=confidence,
                corrected_label=corrected_label,
            )
            st.success(f"Saved to data/collected/{label}/")

        pending.append((result["crop"], label, file.name, predicted, confidence, corrected_label))
        st.divider()

    if pending and st.button(f"Save all {len(pending)} image(s) above", disabled=not consent):
        for crop, label, filename, predicted, confidence, corrected_label in pending:
            _save_and_log(
                crop, label, filename, source="image_upload", original_filename=filename,
                model_used=model_name, predicted_emotion=predicted, confidence=confidence,
                corrected_label=corrected_label,
            )
        st.success(f"Saved {len(pending)} image(s).")


def _render_video_tab(model, model_family: str, img_size: int, detect_fn, model_name: str, consent: bool) -> None:
    st.subheader("Upload a video")
    video_file = st.file_uploader("Video (MP4/MOV/AVI)", type=["mp4", "mov", "avi"])
    every_n = st.number_input("Sample every N frames", min_value=1, value=15, step=1)
    if video_file is None:
        st.info("Upload a video to sample frames from it.")
        return
    if not st.button("Process video"):
        return
    if not consent:
        st.warning("Consent isn't checked, so frames will be analysed but nothing will be saved.")

    with tempfile.NamedTemporaryFile(suffix=Path(video_file.name).suffix, delete=False) as tmp:
        tmp.write(video_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        probe = cv2.VideoCapture(str(tmp_path))
        total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        probe.release()
        if not total_frames:
            st.error(VIDEO_SUFFIX_ERROR)
            return

        progress = st.progress(0.0)
        bar_placeholder, line_placeholder = st.empty(), st.empty()
        emotion_counts = {c: 0 for c in CLASS_NAMES}
        top_emotion_indices: List[int] = []
        n_sampled = n_saved = 0

        for frame_index, frame in sample_video_frames(tmp_path, int(every_n)):
            n_sampled += 1
            result = predict_face(frame, detect_fn, model, model_family, img_size, REALTIME["face_padding"])
            if result is not None:
                probs = result["probs"]
                top_idx = int(np.argmax(probs))
                predicted, confidence = CLASS_NAMES[top_idx], float(probs[top_idx])
                emotion_counts[predicted] += 1
                top_emotion_indices.append(top_idx)

                if consent:
                    _save_and_log(
                        result["crop"], predicted, f"frame{frame_index}", source="video_upload",
                        original_filename=video_file.name, model_used=model_name,
                        predicted_emotion=predicted, confidence=confidence, corrected_label="",
                    )
                    n_saved += 1

                bar_placeholder.bar_chart(pd.Series(emotion_counts, name="count"))
                line_placeholder.line_chart(pd.Series(top_emotion_indices, name="top class index"))
            progress.progress(min(frame_index / total_frames, 1.0))
    finally:
        tmp_path.unlink(missing_ok=True)

    progress.progress(1.0)
    st.write(f"Sampled {n_sampled} frame(s) (every {every_n}); saved {n_saved}.")
    st.caption(
        "No per-frame correction UI here (too slow for video): filenames carry the model's "
        "predicted emotion, which is a sorting hint for manual review, not verified ground truth."
    )
    st.caption("Top-class-index legend: " + ", ".join(f"{i}={c}" for i, c in enumerate(CLASS_NAMES)))


def _render_analytics_tab() -> None:
    st.subheader("Collected data so far")
    if not LOG_PATH.exists():
        st.info(f"No data collected yet ({LOG_PATH} doesn't exist). Save something from the "
                "Image or Video tab first.")
        return

    df = pd.read_csv(LOG_PATH)
    if df.empty:
        st.info("The log file exists but is empty.")
        return

    st.metric("Total collected", len(df))
    label = df["corrected_label"].where(df["corrected_label"].fillna("") != "", df["predicted_emotion"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Collected class distribution")
        st.bar_chart(label.value_counts().reindex(CLASS_NAMES, fill_value=0))
    with col_b:
        st.write("Original FER2013 training distribution (for comparison)")
        fer_counts = fer2013_train_class_counts()
        if fer_counts is None:
            st.info("FER2013 training data not found locally (data/raw/fer2013/train) - skipping.")
        else:
            st.bar_chart(pd.Series(fer_counts).reindex(CLASS_NAMES, fill_value=0))
            st.caption("Compare the two charts to see which classes (often *disgust*) still need more real examples.")

    st.write("Filter")
    sources = sorted(df["source"].unique())
    source_filter = st.multiselect("Source", sources, default=sources)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    min_date, max_date = df["timestamp"].min().date(), df["timestamp"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    filtered = df[df["source"].isin(source_filter)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[(filtered["timestamp"].dt.date >= start) & (filtered["timestamp"].dt.date <= end)]

    st.write("Most recent 20 entries")
    st.dataframe(filtered.sort_values("timestamp", ascending=False).head(20))


def main() -> None:
    """Build the sidebar (model choice + consent) and the three data-collection tabs."""
    set_seed(SEED)
    ensure_dirs()
    st.set_page_config(page_title="Driver Emotion - Data Collection", layout="wide")
    st.title("Driver Emotion Recognition - Data Collection")
    st.caption(
        "A tool for building a labelled dataset from real photos/video - not a live webcam demo. "
        "Every save uses the same face detection and preprocessing as training."
    )

    with st.sidebar:
        st.header("Settings")
        model_names = list_available_models()
        if not model_names:
            st.error(f"No .keras models found in {MODELS_DIR}/. Add one and reload the page.")
            st.stop()
        selected = st.selectbox("Model for inference", model_names, index=0)
        consent = st.checkbox(
            "I have consent to store these photos/frames for research use.", value=False,
            help="Every save button on every tab stays disabled until this is checked.",
        )
        if not consent:
            st.warning("Saving is disabled until consent is confirmed.")

    model_path = MODELS_DIR / f"{selected}.keras"
    try:
        model, img_size = _cached_model(str(model_path))
        model_family = infer_model_family(model)
    except Exception as exc:  # noqa: BLE001 - surface any load/architecture problem in the UI, not a crash
        st.error(f"Could not load '{selected}': {exc}")
        st.stop()
    detect_fn = _cached_detector()

    tab_images, tab_video, tab_analytics = st.tabs(["Image Upload", "Video Upload", "Analytics Dashboard"])
    with tab_images:
        _render_image_tab(model, model_family, img_size, detect_fn, selected, consent)
    with tab_video:
        _render_video_tab(model, model_family, img_size, detect_fn, selected, consent)
    with tab_analytics:
        _render_analytics_tab()


if __name__ == "__main__":
    main()
