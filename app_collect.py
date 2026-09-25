"""Streamlit app: driver emotion recognition demo + data-collection tool.

Analyses a face from an uploaded image, an uploaded video, or a live browser webcam feed
(via streamlit-webrtc), with traffic-scenario simulation (reusing the exact corruption
functions from the project's own robustness study, src/robustness.py's CONDITION_FUNCS) and
a simple risk/safety heuristic layered on top of the model's prediction. The Image and Video
tabs can also save the faces they detect as labelled training data.

Every prediction goes through the exact same face detection and preprocessing as training
(`src/preprocess.py`) and the desktop live app (`src/realtime.py`'s `load_model_for_inference`
/`predict_face`, imported here rather than duplicated).

Run:
    streamlit run app_collect.py

Nothing is written to disk on any tab until the sidebar consent checkbox is ticked, and the
Real-time video tab never saves anything at all (it is a live view only).
"""
import csv
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from config import CLASS_NAMES, FER_DIR, MODELS_DIR, NUM_CLASSES, REALTIME, RESULTS_DIR, ROOT, SEED
from src.realtime import build_face_detector, load_model_for_inference, predict_face
from src.robustness import CONDITION_FUNCS
from src.utils import ensure_dirs, set_seed

COLLECTED_DIR = ROOT / "data" / "collected"
LOG_PATH = RESULTS_DIR / "collected_log.csv"
LOG_FIELDS = [
    "timestamp", "source", "original_filename", "model_used",
    "predicted_emotion", "confidence", "corrected_label", "saved_path",
]
KEEP_PREDICTION = "(use model prediction)"
VIDEO_SUFFIX_ERROR = "Could not read this video - check it isn't corrupted and try a common format (MP4/MOV/AVI)."

# One badge color per class (Streamlit's named semantic colors - see .streamlit/config.toml,
# where each is mapped to the same vivid hue used in every chart's chartCategoricalColors).
EMOTION_BADGE_COLOR = {
    "angry": "red", "disgust": "green", "fear": "violet", "happy": "yellow",
    "neutral": "gray", "sad": "blue", "surprise": "orange",
}

# ------------------------------------------------------------------- traffic scenarios
# Each scenario re-applies one condition from the project's own robustness study rather than
# being decorative styling - src.robustness.CONDITION_FUNCS is the exact function that produced
# the numbers in paper/paper.md's Table 7, reused here rather than duplicated.
SCENARIOS = [
    "Normal Urban Driving", "Congested Traffic", "Highway/Freeway",
    "Intersection", "Parking Maneuver", "Night Driving",
]
SCENARIO_TO_CONDITION = {
    "Normal Urban Driving": "clean",
    "Congested Traffic": "occlusion",
    "Highway/Freeway": "motion_blur",
    "Intersection": "head_pose",
    "Parking Maneuver": "glare",
    "Night Driving": "low_light",
}
SCENARIO_EXPLANATION = {
    "Normal Urban Driving": "No simulated degradation - this is the clean-image baseline.",
    "Congested Traffic": "Simulates a hand, phone or sunglasses covering about a fifth of the face.",
    "Highway/Freeway": "Simulates motion and vibration blur from vehicle speed.",
    "Intersection": "Simulates the driver glancing left/right, up to ±20° off-camera.",
    "Parking Maneuver": "Simulates glare from low sun or reflections off glass.",
    "Night Driving": "Simulates low light with sensor noise, as in a night drive or tunnel.",
}
# Accuracy lost (percentage points, custom CNN, full FER2013 test set) under each condition -
# the project's own measured numbers (paper/paper.md Table 7). Used as a fallback wherever
# results/robustness.csv isn't present locally (e.g. a stripped-down handoff copy of the app).
PUBLISHED_ACCURACY_LOST = {
    "clean": 0.0, "low_light": 39.0, "glare": 16.1, "motion_blur": 23.5,
    "occlusion": 23.2, "head_pose": 0.8,
}

# A simple, documented heuristic - NOT a validated safety claim. See paper/paper.md Section 6:
# treat every prediction as a weak, probabilistic cue, never as a fact about the driver.
RISK_LEVEL = {
    "angry": "HIGH", "fear": "HIGH", "sad": "MEDIUM", "disgust": "MEDIUM",
    "surprise": "MEDIUM", "neutral": "LOW", "happy": "LOW",
}
RISK_BADGE_COLOR = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
SAFETY_RECOMMENDATION = {
    "angry": "Consider pulling over safely and taking a moment before continuing.",
    "fear": "If something startled you, slow down and reassess before continuing.",
    "sad": "Reduced alertness has been linked to low mood - stay extra attentive to the road.",
    "disgust": "No specific driving concern - stay focused on the road.",
    "surprise": "Stay alert for whatever caused it until it passes.",
    "neutral": "No specific concern.",
    "happy": "No specific concern.",
}


@dataclass
class DetectionSettings:
    """The sidebar's scenario/threshold/display-toggle choices, threaded through every tab."""
    scenario: str
    confidence_threshold: float
    update_frequency: int
    show_face_box: bool
    show_emotion_text: bool
    show_risk: bool
    show_scenario_info: bool


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


def model_class_names(model) -> List[str]:
    """The class name for each of `model`'s output units, in order, from its output shape.

    7 units -> CLASS_NAMES (FER2013, includes "neutral"). 6 units -> CLASS_NAMES minus
    "neutral" - a KMU-FED fine-tuned model (src/kmu_fed_data.py, config.KMU_FED), whose 6
    classes correspond 1:1, in the same alphabetical order, to FER2013's classes minus
    neutral (AN=angry, DI=disgust, FE=fear, HA=happy, SA=sad, SU=surprise). This is what lets
    a 6-class model drop straight into every EMOTION_BADGE_COLOR/RISK_LEVEL/
    SAFETY_RECOMMENDATION lookup below unchanged: they're all keyed by these same names, and
    a 6-class model's predictions are always a valid subset of them.
    """
    units = model.output_shape[-1]
    if units == NUM_CLASSES:
        return CLASS_NAMES
    if units == NUM_CLASSES - 1:
        return [c for c in CLASS_NAMES if c != "neutral"]
    raise ValueError(
        f"No class-name mapping is defined for a model with {units} output units "
        f"(expected {NUM_CLASSES} or {NUM_CLASSES - 1})."
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


def scenario_condition(scenario: str) -> str:
    """Map a traffic-scenario label to its src.robustness condition name."""
    return SCENARIO_TO_CONDITION[scenario]


def apply_scenario_corruption(image_bgr: np.ndarray, scenario: str, seed: Optional[int] = None) -> np.ndarray:
    """Apply the scenario's simulated degradation to a BGR image/frame.

    Reuses `src.robustness.CONDITION_FUNCS` - the exact functions used to measure the
    robustness numbers in the paper - rather than duplicating the corruption logic. `seed=None`
    (the default, for an interactive preview) draws a fresh random corruption each call; pass a
    fixed seed for a reproducible one-off preview.
    """
    condition = scenario_condition(scenario)
    rng = np.random.default_rng(seed)
    return CONDITION_FUNCS[condition](image_bgr, rng)


def compute_risk(emotion: str) -> str:
    """Return "LOW"/"MEDIUM"/"HIGH" for one predicted emotion (see RISK_LEVEL's docstring note)."""
    return RISK_LEVEL[emotion]


def is_emotion_acceptable(emotion: str) -> bool:
    """False for the two "alert" emotions (angry, fear) that src/realtime.py also flags."""
    return RISK_LEVEL[emotion] != "HIGH"


def scenario_accuracy_lost(scenario: str, robustness_csv: Path = RESULTS_DIR / "robustness.csv") -> Optional[float]:
    """Percentage points of custom-CNN accuracy lost under this scenario's condition.

    Reads `results/robustness.csv` live if it exists (so this always reflects the most recent
    run), else falls back to the published paper figures in PUBLISHED_ACCURACY_LOST.
    """
    condition = scenario_condition(scenario)
    path = Path(robustness_csv)
    if path.exists():
        try:
            df = pd.read_csv(path).set_index("condition")
            if condition in df.index and "clean" in df.index:
                clean_acc = df.loc["clean", "custom_cnn_accuracy"]
                cond_acc = df.loc[condition, "custom_cnn_accuracy"]
                return round((clean_acc - cond_acc) * 100, 1)
        except Exception:
            pass
    return PUBLISHED_ACCURACY_LOST.get(condition)


def scenario_complexity(scenario: str) -> str:
    """"LOW"/"MEDIUM"/"HIGH", from how much accuracy the custom CNN loses under this scenario."""
    lost = scenario_accuracy_lost(scenario)
    if lost is None or lost < 5:
        return "LOW"
    if lost < 20:
        return "MEDIUM"
    return "HIGH"


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


def _draw_overlay(img: np.ndarray, box: Tuple[int, int, int, int], emotion: str, confidence: float,
                   risk: str, scenario: str, settings: DetectionSettings) -> None:
    """Draw the box/emotion/risk/scenario overlay onto `img` in place, per the display toggles."""
    x, y, w, h = box
    if settings.show_face_box:
        cv2.rectangle(img, (x, y), (x + w, y + h), (139, 92, 246), 2)
    y_text = max(20, y - 10)
    if settings.show_emotion_text:
        cv2.putText(img, f"{emotion.upper()}: {confidence * 100:.2f}%", (x, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_text += 26
    if settings.show_risk:
        cv2.putText(img, f"Risk: {risk}", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y_text += 22
    if settings.show_scenario_info:
        cv2.putText(img, f"Scenario: {scenario.upper()}", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)


def _record_session_stat(emotion: str) -> None:
    """Bump this browser session's running analyzed-count and per-emotion tally (sidebar)."""
    stats = st.session_state.session_stats
    stats["analyzed"] += 1
    stats["emotion_counts"][emotion] += 1


# --------------------------------------------------------------------------- Streamlit UI

@st.cache_resource(show_spinner="Loading model...")
def _cached_model(model_path: str):
    return load_model_for_inference(model_path)


@st.cache_resource(show_spinner="Setting up face detector...")
def _cached_detector():
    return build_face_detector(REALTIME["min_face_confidence"])


def _render_image_tab(
    model, model_family: str, img_size: int, detect_fn, model_name: str, consent: bool,
    settings: DetectionSettings,
) -> None:
    st.subheader("Upload images", icon=":material/photo_camera:")
    class_names = model_class_names(model)
    files = st.file_uploader("Images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if not files:
        st.info("Upload one or more images to run inference.")
        return

    pending = []  # (crop, label, filename, predicted, confidence, corrected_label) for "Save all"
    for i, file in enumerate(files):
        with st.container(border=True):
            image_bgr = _decode_uploaded_image(file)
            if image_bgr is None:
                st.warning(f"**{file.name}** - could not read this file as an image, skipped.")
                continue

            # Inference always runs once on the clean image (its crop is what gets saved, so a
            # scenario preview never pollutes the training set with an artificially degraded
            # photo), plus a second pass on the scenario-corrupted copy when one is selected -
            # that second pass is what's *shown*, since the whole point is to demonstrate how
            # the model's own behaviour changes under that condition.
            clean_result = predict_face(image_bgr, detect_fn, model, model_family, img_size, REALTIME["face_padding"])
            if clean_result is None:
                st.warning(f"**{file.name}** - no face detected, skipped.")
                continue

            preview_bgr, display_result = image_bgr, clean_result
            if settings.scenario != "Normal Urban Driving":
                preview_bgr = apply_scenario_corruption(image_bgr, settings.scenario)
                corrupted = predict_face(preview_bgr, detect_fn, model, model_family, img_size, REALTIME["face_padding"])
                if corrupted is not None:
                    display_result = corrupted
                else:
                    st.caption("The face detector couldn't find a face in the scenario-corrupted "
                               "preview - showing the clean-image result instead.")
                    preview_bgr = image_bgr

            x, y, w, h = display_result["box"]
            probs = display_result["probs"]
            top_idx = int(np.argmax(probs))
            predicted, confidence = class_names[top_idx], float(probs[top_idx])
            confident = confidence >= settings.confidence_threshold
            _record_session_stat(predicted)

            annotated = preview_bgr.copy()
            risk = compute_risk(predicted)
            if confident:
                _draw_overlay(annotated, (x, y, w, h), predicted, confidence, risk, settings.scenario, settings)
            elif settings.show_face_box:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (148, 163, 184), 2)

            st.markdown(f"**{file.name}**")
            if settings.show_scenario_info and settings.scenario != "Normal Urban Driving":
                st.caption(f"Scenario preview: *{settings.scenario}* - {SCENARIO_EXPLANATION[settings.scenario]}")

            if settings.show_emotion_text:
                if confident:
                    st.badge(f"{predicted} · {confidence * 100:.0f}%", color=EMOTION_BADGE_COLOR[predicted])
                else:
                    st.badge(
                        f"Uncertain · {confidence * 100:.0f}% (below {settings.confidence_threshold:.0%} threshold)",
                        color="gray",
                    )
            if settings.show_risk and confident:
                st.badge(f"Risk: {risk}", color=RISK_BADGE_COLOR[risk])

            col_img, col_chart = st.columns(2)
            with col_img:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            with col_chart:
                st.bar_chart(pd.Series(probs, index=class_names, name="probability"))

            if confident:
                with st.container(border=True):
                    st.markdown("**Scenario-specific analysis**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if is_emotion_acceptable(predicted):
                            st.success("Acceptable emotion for this scenario", icon=":material/check_circle:")
                        else:
                            st.warning("Elevated-risk emotion for this scenario", icon=":material/warning:")
                    with col_b:
                        st.metric("Scenario complexity", scenario_complexity(settings.scenario))
                    st.caption(f"Safety note: {SAFETY_RECOMMENDATION[predicted]}")

            correction = st.selectbox("Correct label if wrong", [KEEP_PREDICTION] + CLASS_NAMES, key=f"correct_{i}")
            corrected_label = "" if correction == KEEP_PREDICTION else correction
            label = corrected_label or predicted

            if st.button("Save to dataset", icon=":material/save:", key=f"save_{i}", disabled=not consent):
                _save_and_log(
                    clean_result["crop"], label, file.name, source="image_upload", original_filename=file.name,
                    model_used=model_name, predicted_emotion=predicted, confidence=confidence,
                    corrected_label=corrected_label,
                )
                st.success(f"Saved to data/collected/{label}/", icon=":material/check_circle:")

            pending.append((clean_result["crop"], label, file.name, predicted, confidence, corrected_label))

    if pending and st.button(
        f"Save all {len(pending)} image(s) above", icon=":material/save:", type="primary", disabled=not consent,
    ):
        for crop, label, filename, predicted, confidence, corrected_label in pending:
            _save_and_log(
                crop, label, filename, source="image_upload", original_filename=filename,
                model_used=model_name, predicted_emotion=predicted, confidence=confidence,
                corrected_label=corrected_label,
            )
        st.success(f"Saved {len(pending)} image(s).", icon=":material/check_circle:")


def _render_video_tab(
    model, model_family: str, img_size: int, detect_fn, model_name: str, consent: bool,
    settings: DetectionSettings,
) -> None:
    st.subheader("Upload a video", icon=":material/videocam:")
    class_names = model_class_names(model)
    video_file = st.file_uploader("Video (MP4/MOV/AVI)", type=["mp4", "mov", "avi"])
    every_n = st.number_input("Sample every N frames", min_value=1, value=15, step=1)
    if video_file is None:
        st.info("Upload a video to sample frames from it.")
        return
    if not st.button("Process video", icon=":material/play_arrow:", type="primary"):
        return
    if not consent:
        st.warning("Consent isn't checked, so frames will be analysed but nothing will be saved.")
    if settings.scenario != "Normal Urban Driving":
        st.caption(f"Scenario preview: *{settings.scenario}* - {SCENARIO_EXPLANATION[settings.scenario]} "
                   "Each sampled frame is analysed twice (clean, for saving, and scenario-corrupted, "
                   "for the stats below), so this runs slower than Normal Urban Driving.")

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
        col_bar, col_line = st.columns(2)
        with col_bar:
            bar_card = st.container(border=True)
            bar_card.markdown("**Emotion distribution so far**")
            bar_placeholder = bar_card.empty()
        with col_line:
            line_card = st.container(border=True)
            line_card.markdown("**Top emotion over time**")
            line_placeholder = line_card.empty()

        emotion_counts = {c: 0 for c in CLASS_NAMES}
        top_emotion_indices: List[int] = []
        n_sampled = n_saved = n_uncertain = 0

        for frame_index, frame in sample_video_frames(tmp_path, int(every_n)):
            n_sampled += 1
            clean_result = predict_face(frame, detect_fn, model, model_family, img_size, REALTIME["face_padding"])
            if clean_result is None:
                progress.progress(min(frame_index / total_frames, 1.0))
                continue

            display_result = clean_result
            if settings.scenario != "Normal Urban Driving":
                corrupted_frame = apply_scenario_corruption(frame, settings.scenario)
                corrupted = predict_face(corrupted_frame, detect_fn, model, model_family, img_size, REALTIME["face_padding"])
                if corrupted is not None:
                    display_result = corrupted

            probs = display_result["probs"]
            top_idx = int(np.argmax(probs))
            predicted, confidence = class_names[top_idx], float(probs[top_idx])

            if confidence >= settings.confidence_threshold:
                emotion_counts[predicted] += 1
                top_emotion_indices.append(top_idx)
                _record_session_stat(predicted)

                if consent:
                    _save_and_log(
                        clean_result["crop"], predicted, f"frame{frame_index}", source="video_upload",
                        original_filename=video_file.name, model_used=model_name,
                        predicted_emotion=predicted, confidence=confidence, corrected_label="",
                    )
                    n_saved += 1
                bar_placeholder.bar_chart(pd.Series(emotion_counts, name="count"))
                line_placeholder.line_chart(pd.Series(top_emotion_indices, name="top class index"))
            else:
                n_uncertain += 1
            progress.progress(min(frame_index / total_frames, 1.0))
    finally:
        tmp_path.unlink(missing_ok=True)

    progress.progress(1.0)
    with st.container(horizontal=True):
        st.metric("Frames sampled", n_sampled, border=True)
        st.metric("Frames saved", n_saved, border=True)
        st.metric("Below confidence threshold", n_uncertain, border=True)
    st.caption(
        "No per-frame correction UI here (too slow for video): filenames carry the model's "
        "predicted emotion, which is a sorting hint for manual review, not verified ground truth."
    )
    st.caption("Top-class-index legend: " + ", ".join(f"{i}={c}" for i, c in enumerate(CLASS_NAMES)))


class _LiveStats:
    """Thread-safe box for the real-time tab's live stats.

    `webrtc_streamer`'s video_frame_callback runs on a background WebRTC thread, not the main
    Streamlit script thread, so writing straight to st.session_state from inside it is not
    safe. This tiny lock-protected object is the standard streamlit-webrtc pattern for sharing
    a value from that thread back to the main script, which reads it via `snapshot()`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frame_count = 0
        self.emotion: Optional[str] = None
        self.confidence: float = 0.0
        self.risk: Optional[str] = None

    def update(self, emotion: str, confidence: float, risk: str) -> None:
        with self._lock:
            self.frame_count += 1
            self.emotion, self.confidence, self.risk = emotion, confidence, risk

    def snapshot(self) -> Tuple[int, Optional[str], float, Optional[str]]:
        with self._lock:
            return self.frame_count, self.emotion, self.confidence, self.risk


@st.cache_resource
def _live_stats() -> _LiveStats:
    return _LiveStats()


def _make_realtime_callback(model, model_family: str, img_size: int, detect_fn, settings: DetectionSettings, stats: _LiveStats):
    """Build the per-frame callback for webrtc_streamer: corrupt (if a scenario is active),
    run inference every `update_frequency`-th frame, draw the last known overlay every frame.
    """
    class_names = model_class_names(model)
    state = {"n": 0, "box": None, "emotion": None, "confidence": 0.0, "risk": None}

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        state["n"] += 1

        if settings.scenario != "Normal Urban Driving":
            img = apply_scenario_corruption(img, settings.scenario)

        if state["n"] % settings.update_frequency == 0:
            result = predict_face(img, detect_fn, model, model_family, img_size, REALTIME["face_padding"])
            if result is not None:
                probs = result["probs"]
                top_idx = int(np.argmax(probs))
                emotion, confidence = class_names[top_idx], float(probs[top_idx])
                if confidence >= settings.confidence_threshold:
                    risk = compute_risk(emotion)
                    state.update(box=result["box"], emotion=emotion, confidence=confidence, risk=risk)
                    stats.update(emotion, confidence, risk)

        if state["box"] is not None:
            _draw_overlay(img, state["box"], state["emotion"], state["confidence"], state["risk"],
                          settings.scenario, settings)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return callback


def _render_realtime_tab(model, model_family: str, img_size: int, detect_fn, settings: DetectionSettings) -> None:
    st.subheader("Real-time video analysis", icon=":material/videocam:")
    st.caption(
        "Analyze emotions live from your webcam, with traffic-scenario context. Runs entirely "
        "in this browser tab - nothing here is ever saved to disk."
    )

    stats = _live_stats()
    col_video, col_controls = st.columns([2, 1])

    with col_video:
        ctx = webrtc_streamer(
            key="driver-emotion-realtime",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=_make_realtime_callback(model, model_family, img_size, detect_fn, settings, stats),
            async_processing=True,
        )

    with col_controls:
        with st.container(border=True):
            st.markdown("**Controls**")
            emotion_slot, confidence_slot, risk_slot, frames_slot = st.empty(), st.empty(), st.empty(), st.empty()

    if not ctx.state.playing:
        emotion_slot.metric("Current emotion", "-")
        confidence_slot.metric("Confidence", "-")
        frames_slot.caption("Click \"Start\" above to begin.")
        return

    # Poll the shared stats a few times a second while the stream is live, so this panel
    # updates without the visitor needing to click anything else. The loop exits as soon as
    # the stream does (Stop clicked, or the tab disconnects); the large bound is only a safety
    # net so a genuinely long session isn't cut short, and so this can never hang forever.
    for _ in range(12_000):  # ~1 hour at 0.3s/tick
        if not ctx.state.playing:
            break
        frame_count, emotion, confidence, risk = stats.snapshot()
        emotion_slot.metric("Current emotion", (emotion or "-").upper())
        confidence_slot.metric("Confidence", f"{confidence * 100:.1f}%" if emotion else "-")
        if risk:
            risk_slot.badge(f"Risk: {risk}", color=RISK_BADGE_COLOR[risk])
        frames_slot.caption(f"Frames processed: {frame_count}")
        time.sleep(0.3)


def _render_analytics_tab() -> None:
    st.subheader("Collected data so far", icon=":material/query_stats:")
    if not LOG_PATH.exists():
        st.info(f"No data collected yet ({LOG_PATH} doesn't exist). Save something from the "
                "image or video tab first.")
        return

    df = pd.read_csv(LOG_PATH)
    if df.empty:
        st.info("The log file exists but is empty.")
        return

    label = df["corrected_label"].where(df["corrected_label"].fillna("") != "", df["predicted_emotion"])
    corrected_share = (df["corrected_label"].fillna("") != "").mean() * 100

    with st.container(horizontal=True):
        st.metric("Total collected", len(df), border=True)
        st.metric("Classes represented", label.nunique(), border=True)
        st.metric("Manually corrected", f"{corrected_share:.0f}%", border=True)

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("**Collected class distribution**")
            st.bar_chart(label.value_counts().reindex(CLASS_NAMES, fill_value=0))
    with col_b:
        with st.container(border=True):
            st.markdown("**Original FER2013 training distribution**")
            fer_counts = fer2013_train_class_counts()
            if fer_counts is None:
                st.caption("FER2013 training data not found locally (data/raw/fer2013/train) - skipping.")
            else:
                st.bar_chart(pd.Series(fer_counts).reindex(CLASS_NAMES, fill_value=0))
                st.caption("Compare the two charts to see which classes still need more real examples.")

    with st.container(border=True):
        st.markdown("**Filter**", help="Narrows the table below only.")
        col_source, col_dates = st.columns(2)
        with col_source:
            sources = sorted(df["source"].unique())
            source_filter = st.multiselect("Source", sources, default=sources)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        min_date, max_date = df["timestamp"].min().date(), df["timestamp"].max().date()
        with col_dates:
            date_range = st.date_input(
                "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
            )

        filtered = df[df["source"].isin(source_filter)]
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            filtered = filtered[(filtered["timestamp"].dt.date >= start) & (filtered["timestamp"].dt.date <= end)]

        st.markdown("**Most recent 20 entries**")
        st.dataframe(filtered.sort_values("timestamp", ascending=False).head(20))


def _render_about_tab() -> None:
    st.subheader("About this project", icon=":material/info:")
    st.markdown(
        "This tool analyses a driver's facial expression from an image, a video, or a live "
        "webcam feed, using one of two models trained from scratch on FER2013: a compact "
        "custom CNN and a fine-tuned VGG16."
    )
    with st.container(border=True):
        st.markdown("**Measured results (full FER2013 test set, 7,178 images)**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Custom CNN accuracy", "67.5%", border=True)
            st.metric("Custom CNN speed", "~25 FPS (CPU)", border=True)
        with col_b:
            st.metric("VGG16 accuracy", "65.8%", border=True)
            st.metric("VGG16 speed", "~4.6 FPS (CPU)", border=True)
        st.caption(
            "The accuracy gap is small but statistically significant (paired bootstrap 95% CI "
            "+0.7 to +2.8 points; exact McNemar p = 0.0014)."
        )
    st.markdown(
        "**Traffic scenarios** simulate driving conditions from the project's own robustness "
        "study rather than being decorative: each one reapplies the exact corruption function "
        "used to measure how much accuracy the models lose in that condition."
    )
    st.markdown(
        "**Data collection.** The Image and Video tabs can optionally save the faces they "
        "detect to `data/collected/`, gated on the sidebar consent checkbox - nothing is saved "
        "by default, and nothing from the Real-time video tab is ever saved."
    )
    st.caption(
        "This is a research prototype, not a certified safety system. Treat every prediction, "
        "risk level and recommendation here as a weak, probabilistic cue - never as a fact "
        "about the driver."
    )


def main() -> None:
    """Build the sidebar (scenario, detection settings, model choice, consent) and the tabs."""
    set_seed(SEED)
    ensure_dirs()
    st.set_page_config(page_title="Driver emotion - data collection", page_icon=":material/mood:", layout="wide")
    st.title("Driver emotion recognition system", icon=":material/mood:")
    st.caption(
        "AI-powered emotion detection for multiple traffic scenarios. Every save uses the "
        "same face detection and preprocessing as training."
    )

    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {"analyzed": 0, "emotion_counts": {c: 0 for c in CLASS_NAMES}}

    with st.sidebar:
        st.header("Settings", icon=":material/tune:")

        st.subheader("Traffic scenario", icon=":material/directions_car:")
        scenario = st.selectbox("Select current scenario", SCENARIOS, index=0)
        with st.expander("How this scenario works differently"):
            st.write(SCENARIO_EXPLANATION[scenario])
            lost = scenario_accuracy_lost(scenario)
            if lost:
                st.caption(f"Measured effect: the custom CNN lost about {lost:.1f} accuracy "
                           "points under this condition in the project's own robustness study.")
            else:
                st.caption("No measured effect (this is the clean-image baseline).")

        st.subheader("Detection settings", icon=":material/settings_input_component:")
        update_frequency = st.slider(
            "Update frequency (frames)", 1, 10, 5,
            help="Real-time video tab only: run full inference once every N frames.",
        )
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.30, step=0.05)

        st.subheader("Display options", icon=":material/visibility:")
        show_face_box = st.checkbox("Show face detection box", value=True)
        show_emotion_text = st.checkbox("Show emotion text", value=True)
        show_risk = st.checkbox("Show risk indicator", value=True)
        show_scenario_info = st.checkbox("Show scenario info", value=True)

        settings = DetectionSettings(
            scenario, confidence_threshold, update_frequency,
            show_face_box, show_emotion_text, show_risk, show_scenario_info,
        )

        st.subheader("Model", icon=":material/memory:")
        model_names = list_available_models()
        if not model_names:
            st.error(f"No .keras models found in {MODELS_DIR}/. Add one and reload the page.")
            st.stop()
        selected = st.selectbox("Model for inference", model_names, index=0)
        consent = st.checkbox(
            "I have consent to store these photos/frames for research use.", value=False,
            help="Every save button on every tab stays disabled until this is checked.",
        )
        if consent:
            st.badge("Saving enabled", icon=":material/lock_open:", color="green")
        else:
            st.badge("Saving disabled", icon=":material/lock:", color="red")

        st.subheader("Session statistics", icon=":material/bar_chart:")
        stats = st.session_state.session_stats
        st.metric("Images/frames analyzed", stats["analyzed"])
        if stats["analyzed"]:
            st.bar_chart(pd.Series(stats["emotion_counts"]).reindex(CLASS_NAMES, fill_value=0), height=150)

    model_path = MODELS_DIR / f"{selected}.keras"
    try:
        model, img_size = _cached_model(str(model_path))
        model_family = infer_model_family(model)
    except Exception as exc:  # noqa: BLE001 - surface any load/architecture problem in the UI, not a crash
        st.error(f"Could not load '{selected}': {exc}")
        st.stop()
    detect_fn = _cached_detector()
    with st.sidebar:
        st.badge(f"Preprocessing: {model_family}", icon=":material/memory:", color="violet")

    tab_images, tab_realtime, tab_video, tab_analytics, tab_about = st.tabs([
        ":material/photo_camera: Image upload",
        ":material/videocam: Real-time video",
        ":material/upload_file: Video upload",
        ":material/query_stats: Analytics dashboard",
        ":material/info: About",
    ])
    with tab_images:
        _render_image_tab(model, model_family, img_size, detect_fn, selected, consent, settings)
    with tab_realtime:
        _render_realtime_tab(model, model_family, img_size, detect_fn, settings)
    with tab_video:
        _render_video_tab(model, model_family, img_size, detect_fn, selected, consent, settings)
    with tab_analytics:
        _render_analytics_tab()
    with tab_about:
        _render_about_tab()


if __name__ == "__main__":
    main()
