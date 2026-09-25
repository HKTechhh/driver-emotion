"""Real-time driver emotion recognition from a webcam (or video file).

Run as a module from the repo root, e.g.:
    python -m src.realtime --model custom_cnn --model-path models/cnn_v1.keras --log results/live_log.csv

Press 'q' to quit.

Face detection prefers MediaPipe's Tasks API (downloading the small BlazeFace model on
first use); if that's unavailable for any reason (no mediapipe, no network, API
mismatch), it falls back to OpenCV's bundled Haar cascade, per PLAN.md's Phase 0/7
contingency. Preprocessing is shared with training via src/preprocess.py so a live frame
is treated exactly like a training image.

By default no image is ever saved, only the CSV log (timestamp, predicted emotion,
confidence, FPS) - see Section 6 of paper/paper.md. Pass --save-frames to additionally
write each detected face crop to disk (e.g. to grow the training set); this is opt-in and
was NOT used for any of the results reported in the paper. Get informed consent from
whoever is on camera before using it (docs/car_deployment_guide.md, Section 8).

`load_model_for_inference` and `predict_face` hold the single-frame detect/preprocess/predict
path with no temporal smoothing, drawing or logging; `process_frame` (below) adds those for
this live app, and `app_collect.py` (Streamlit data-collection tool) imports the same two
functions directly, so both stay in exact sync with training's preprocessing.
"""
import argparse
import csv
import platform
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, MODELS_DIR, REALTIME, RESULTS_DIR
from src.preprocess import preprocess
from src.utils import ensure_dirs

BLAZEFACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
BLAZEFACE_PATH = MODELS_DIR / "blaze_face_short_range.tflite"

HAAR_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)
HAAR_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"

Box = Tuple[int, int, int, int, float]  # x, y, w, h, confidence
DetectFn = Callable[[np.ndarray], List[Box]]


class EmotionTracker:
    """Smooths per-frame class probabilities and tracks how long an alert emotion has led."""

    def __init__(self, smoothing_window: int, alert_emotions: Tuple[str, ...], alert_seconds: float):
        self.buffer: deque = deque(maxlen=smoothing_window)
        self.alert_emotions = alert_emotions
        self.alert_seconds = alert_seconds
        self._alert_emotion: Optional[str] = None
        self._alert_start: Optional[float] = None

    def update(self, probs: np.ndarray, now: float) -> Tuple[np.ndarray, str, float, bool]:
        """Add one frame's probabilities; return (smoothed_probs, top_emotion, confidence, alert_triggered)."""
        self.buffer.append(probs)
        smoothed = np.mean(self.buffer, axis=0)
        top_idx = int(np.argmax(smoothed))
        top_emotion = CLASS_NAMES[top_idx]
        confidence = float(smoothed[top_idx])

        if top_emotion in self.alert_emotions:
            if top_emotion != self._alert_emotion:
                self._alert_emotion = top_emotion
                self._alert_start = now
            triggered = (now - self._alert_start) >= self.alert_seconds
        else:
            self._alert_emotion = None
            self._alert_start = None
            triggered = False

        return smoothed, top_emotion, confidence, triggered


def _download_file(url: str, dest: Path, timeout: int = 15) -> bool:
    """Best-effort download; returns False (never raises) on any failure."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as response, open(dest, "wb") as f:
            f.write(response.read())
        return True
    except Exception as exc:
        print(f"Could not download {url}: {exc}")
        return False


def _build_mediapipe_detector(min_confidence: float) -> DetectFn:
    """Build a MediaPipe Tasks FaceDetector. Raises on any failure; caller falls back to Haar."""
    import mediapipe as mp

    if not BLAZEFACE_PATH.exists() and not _download_file(BLAZEFACE_URL, BLAZEFACE_PATH):
        raise RuntimeError("BlazeFace model unavailable (no local copy, download failed)")

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(BLAZEFACE_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=min_confidence,
    )
    detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect(frame_bgr: np.ndarray) -> List[Box]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        boxes = []
        for det in result.detections:
            bbox = det.bounding_box
            score = det.categories[0].score if det.categories else 0.0
            boxes.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height, float(score)))
        return boxes

    return detect


def _build_haar_detector(min_confidence: float) -> DetectFn:
    """OpenCV Haar cascade fallback.

    Some opencv-python builds (e.g. the opencv-contrib wheel used here) ship no cascade
    data files at all under `cv2.data.haarcascades`, so this downloads the standard
    frontal-face cascade from OpenCV's own GitHub repo if a local copy isn't cached yet.
    """
    bundled_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade_path = bundled_path if bundled_path.exists() else HAAR_CASCADE_PATH
    if not cascade_path.exists() and not _download_file(HAAR_CASCADE_URL, HAAR_CASCADE_PATH):
        raise RuntimeError("Haar cascade unavailable (not bundled, and download failed)")

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Haar cascade at {cascade_path} failed to load")

    def detect(frame_bgr: np.ndarray) -> List[Box]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [(int(x), int(y), int(w), int(h), 1.0) for x, y, w, h in faces]

    return detect


def build_face_detector(min_confidence: float) -> DetectFn:
    """MediaPipe Face Detection if available, else OpenCV's Haar cascade."""
    try:
        detect = _build_mediapipe_detector(min_confidence)
        print("Face detector: MediaPipe (BlazeFace).")
        return detect
    except Exception as exc:
        print(f"MediaPipe face detector unavailable ({exc}); falling back to OpenCV Haar cascade.")
        return _build_haar_detector(min_confidence)


def _crop_with_padding(frame: np.ndarray, box: Tuple[int, int, int, int], padding: float) -> np.ndarray:
    """Crop `box` out of `frame`, expanded by `padding` fraction on each side, clamped to the frame."""
    x, y, w, h = box
    pad_x, pad_y = int(w * padding), int(h * padding)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(frame.shape[1], x + w + pad_x), min(frame.shape[0], y + h + pad_y)
    return frame[y0:y1, x0:x1]


def _draw_probability_bars(frame: np.ndarray, probs: np.ndarray, origin: Tuple[int, int] = (10, 10)) -> None:
    """Draw a compact bar chart of all class probabilities in the corner of the frame, in place."""
    x0, y0 = origin
    bar_height, bar_max_width = 14, 100
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        y = y0 + i * (bar_height + 4)
        cv2.rectangle(frame, (x0, y), (x0 + bar_max_width, y + bar_height), (60, 60, 60), 1)
        cv2.rectangle(frame, (x0, y), (x0 + int(bar_max_width * prob), y + bar_height), (0, 200, 0), -1)
        cv2.putText(
            frame, f"{name[:4]} {prob:.2f}", (x0 + bar_max_width + 6, y + bar_height - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )


def load_model_for_inference(model_path: str) -> Tuple[tf.keras.Model, int]:
    """Load a saved .keras model and return it with its expected square input size."""
    model = tf.keras.models.load_model(model_path)
    img_size = model.input_shape[1]
    return model, img_size


def predict_face(
    frame_bgr: np.ndarray,
    detect_fn: DetectFn,
    model: tf.keras.Model,
    model_name: str,
    img_size: int,
    face_padding: float,
) -> Optional[dict]:
    """Detect the largest face, crop, preprocess and predict, for one frame - no smoothing,
    drawing or logging (that's `process_frame`, below, for the live app).

    Returns None if no face was found, else a dict: {"box": (x, y, w, h), "crop": the raw BGR
    face crop before grayscale/resize, "probs": per-class probabilities in CLASS_NAMES order}.
    """
    boxes = detect_fn(frame_bgr)
    if not boxes:
        return None

    x, y, w, h, _conf = max(boxes, key=lambda b: b[2] * b[3])
    crop = _crop_with_padding(frame_bgr, (x, y, w, h), face_padding)
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    model_input = preprocess(resized[np.newaxis, ..., np.newaxis], model_name)
    probs = np.asarray(model(model_input, training=False))[0]
    return {"box": (x, y, w, h), "crop": crop, "probs": probs}


def process_frame(
    frame_bgr: np.ndarray,
    detect_fn: DetectFn,
    model: tf.keras.Model,
    model_name: str,
    img_size: int,
    tracker: EmotionTracker,
    fps: float,
    face_padding: float,
) -> Tuple[np.ndarray, Optional[dict], Optional[np.ndarray]]:
    """Detect -> preprocess -> predict -> smooth -> draw, for one frame.

    Returns the annotated frame, a log row dict (or None if no face was found), and the raw
    BGR face crop before grayscale/resize (or None) for callers that want to save it (e.g.
    --save-frames). Contains no cv2.imshow/waitKey calls, so it can be unit-tested on a
    single static frame.
    """
    annotated = frame_bgr.copy()
    result = predict_face(frame_bgr, detect_fn, model, model_name, img_size, face_padding)
    if result is None:
        cv2.putText(
            annotated, "No face detected", (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return annotated, None, None

    x, y, w, h = result["box"]
    crop = result["crop"]
    smoothed, top_emotion, confidence, alert_triggered = tracker.update(result["probs"], time.time())

    banner_height = 40
    min_label_y = (banner_height + 20) if alert_triggered else 20
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label_y = y - 10 if y - 10 > min_label_y else y + 25  # stay clear of the top edge and any alert banner
    cv2.putText(
        annotated, f"{top_emotion} {confidence * 100:.0f}%", (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )
    _draw_probability_bars(annotated, smoothed, origin=(10, banner_height + 10 if alert_triggered else 10))  # keep the bars clear of the alert banner
    cv2.putText(
        annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
    )
    if alert_triggered:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], banner_height), (0, 0, 200), -1)
        cv2.putText(
            annotated, "Stay calm - take a breath", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "emotion": top_emotion,
        "confidence": confidence,
        "fps": fps,
    }
    return annotated, log_row, crop


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for the real-time app."""
    parser = argparse.ArgumentParser(description="Real-time driver emotion recognition.")
    parser.add_argument("--model", choices=["custom_cnn", "vgg16"], required=True)
    parser.add_argument("--model-path", dest="model_path", type=str, required=True)
    parser.add_argument("--camera", type=int, default=REALTIME["camera_index"])
    parser.add_argument("--video", type=str, default=None, help="Video file, instead of a live camera.")
    parser.add_argument("--log", type=str, default=str(RESULTS_DIR / "live_log.csv"))
    parser.add_argument(
        "--save-frames", dest="save_frames", action="store_true",
        help="Also save each detected face crop as a JPEG (default: off, no images are ever saved).",
    )
    parser.add_argument(
        "--frames-dir", dest="frames_dir", type=str, default=None,
        help="Where to save face crops when --save-frames is set (default: '<log>_frames/' next to the log).",
    )
    parser.add_argument(
        "--save-every", dest="save_every", type=int, default=5,
        help="With --save-frames, save one face crop out of every N detected (default: 5, since "
             "consecutive video frames are near-duplicates and add little to a training set).",
    )
    parser.add_argument(
        "--backend", choices=["auto", "dshow", "msmf", "any"], default="auto",
        help="OpenCV camera backend (Windows only affects dshow/msmf; ignored for --video). "
             "'auto' (default) tries the platform default first, then DirectShow, then Media "
             "Foundation, since some Windows webcam drivers only work with one of them. If the "
             "camera window never opens on Windows, try --backend dshow explicitly.",
    )
    return parser


def _open_capture(source, backend: str) -> cv2.VideoCapture:
    """Open `source` (a camera index or a video file path), trying alternate backends on failure.

    OpenCV's default camera backend on Windows (Media Foundation) fails to open some webcams
    outright; DirectShow works for most of those. This only matters for a live camera index
    (a video file always uses its own container-format backend, so `backend` is ignored then).
    """
    is_camera = isinstance(source, int)
    forced = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}.get(backend)
    if not is_camera or forced is not None:
        return cv2.VideoCapture(source, forced) if forced is not None else cv2.VideoCapture(source)

    cap = cv2.VideoCapture(source)  # backend == "auto": platform default first
    if cap.isOpened() or platform.system() != "Windows":
        return cap
    for candidate in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap.release()
        cap = cv2.VideoCapture(source, candidate)
        if cap.isOpened():
            return cap
    return cap  # still not opened; caller reports the error


def main() -> None:
    """Open the camera (or video file), run the live loop, and log every frame's prediction."""
    args = build_arg_parser().parse_args()
    ensure_dirs()

    model, img_size = load_model_for_inference(args.model_path)
    detect_fn = build_face_detector(REALTIME["min_face_confidence"])
    tracker = EmotionTracker(REALTIME["smoothing_window"], REALTIME["alert_emotions"], REALTIME["alert_seconds"])

    source = args.video if args.video else args.camera
    cap = _open_capture(source, args.backend)
    if not cap.isOpened():
        hint = (
            " Nothing else (Teams, Zoom, the Windows Camera app, browser tabs) can be using it at "
            "the same time; try --camera 1 if you have more than one camera; on Windows, try "
            "--backend dshow or --backend msmf explicitly if 'auto' didn't already."
            if not args.video else ""
        )
        raise SystemExit(f"Could not open video source: {source}.{hint}")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "emotion", "confidence", "fps"]

    frames_dir = None
    if args.save_frames:
        frames_dir = Path(args.frames_dir) if args.frames_dir else log_path.parent / f"{log_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        fieldnames.append("frame_file")
        print(f"--save-frames is on: face crops will be written to {frames_dir}/ "
              f"(one in every {args.save_every}). Make sure everyone on camera has consented to this.")

    log_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    prev_time = time.time()
    faces_seen = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            annotated, log_row, face_crop = process_frame(
                frame, detect_fn, model, args.model, img_size, tracker, fps, REALTIME["face_padding"]
            )
            if log_row is not None:
                if frames_dir is not None:
                    faces_seen += 1
                    log_row["frame_file"] = ""
                    if face_crop is not None and faces_seen % args.save_every == 0:
                        # Filename carries the *predicted* emotion, not a verified label - anyone
                        # curating this into a training set still needs to check/correct it by eye.
                        # The running counter (not just the timestamp, which only has 1-second
                        # resolution) keeps filenames unique when several frames land in one second.
                        stamp = log_row["timestamp"].replace(":", "-")
                        filename = f"{stamp}_{faces_seen:06d}_{log_row['emotion']}_{log_row['confidence']:.2f}.jpg"
                        cv2.imwrite(str(frames_dir / filename), face_crop)
                        log_row["frame_file"] = filename
                writer.writerow(log_row)

            cv2.imshow("Driver Emotion Recognition", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()


if __name__ == "__main__":
    main()
