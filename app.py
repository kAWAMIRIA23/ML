"""Waste Classification and Object Detection with YOLOv8 + Gradio."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS_CANDIDATES = (
    ROOT / "weights" / "best.pt",
    ROOT / "best.pt",
    ROOT / "weights" / "best.onnx",
)
# General object detector used to propose regions; waste class comes from best.pt.
DET_WEIGHTS = ROOT / "yolov8n.pt"
BOX_COLOR = (0, 220, 255)  # yellow-ish in BGR for drawing on BGR canvas
TEXT_BG = (0, 180, 220)
TEXT_FG = (0, 0, 0)


def resolve_weights() -> Path:
    for path in WEIGHTS_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No YOLOv8 weights found. Place best.pt (or best.onnx) under ./weights/"
    )


WEIGHTS = resolve_weights()
cls_model = YOLO(str(WEIGHTS))
# Prefer a local detector if present; otherwise Ultralytics downloads yolov8n.pt.
det_model = YOLO(str(DET_WEIGHTS) if DET_WEIGHTS.is_file() else "yolov8n.pt")


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def classify_crop(rgb_crop: np.ndarray) -> tuple[str, float]:
    """Return top waste class name and confidence for an RGB crop."""
    result = cls_model.predict(rgb_crop, verbose=False)[0]
    if result.probs is None:
        # Detection-style classifier weights — use top box if present.
        if result.boxes is not None and len(result.boxes):
            box = result.boxes[0]
            name = result.names[int(box.cls[0])]
            return name, float(box.conf[0])
        return "unknown", 0.0
    name = result.names[int(result.probs.top1)]
    return name, float(result.probs.top1conf)


def draw_label_box(
    bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
) -> None:
    """Draw a yellow bounding box with an uppercase class label (matches demo UI)."""
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), BOX_COLOR, 2)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty1 = max(0, y1 - th - 8)
    ty2 = ty1 + th + 8
    cv2.rectangle(bgr, (x1, ty1), (x1 + tw + 8, ty2), TEXT_BG, -1)
    cv2.putText(
        bgr,
        label,
        (x1 + 4, ty2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_FG,
        2,
        cv2.LINE_AA,
    )


def format_results(all_lines: list[str]) -> str:
    if not all_lines:
        return "No waste objects detected above the confidence threshold."
    return "\n".join(all_lines)


def annotate_rgb(rgb_image: np.ndarray, conf: float) -> tuple[np.ndarray, list[str]]:
    """
    Detect objects, classify each region as a waste type, draw labeled boxes.

    Uses yolov8n for boxes + best.pt (TrashNet classifier) for waste labels so
    the Detected Image shows per-item bounding boxes like METAL 0.87.
    """
    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3:
        return rgb, []

    h, w = rgb.shape[:2]
    lines: list[str] = []
    canvas_bgr = rgb_to_bgr(rgb.copy())

    # If the primary weights are already a detector, use them directly.
    if getattr(cls_model, "task", None) == "detect":
        result = cls_model.predict(rgb, conf=conf, verbose=False)[0]
        plotted = bgr_to_rgb(result.plot())
        if result.boxes is not None:
            for box in result.boxes:
                score = float(box.conf[0])
                if score < conf:
                    continue
                name = result.names[int(box.cls[0])]
                lines.append(f"{name}: {score:.2f}")
        return plotted, lines

    det = det_model.predict(rgb, conf=max(conf, 0.25), verbose=False)[0]
    boxes = det.boxes

    if boxes is not None and len(boxes):
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            # Pad slightly so thin objects still classify well.
            pad = 4
            xa, ya = max(0, x1 - pad), max(0, y1 - pad)
            xb, yb = min(w, x2 + pad), min(h, y2 + pad)
            crop = rgb[ya:yb, xa:xb]
            if crop.size == 0:
                continue

            name, score = classify_crop(crop)
            if score < conf:
                continue

            label = f"{name.upper()} {score:.2f}"
            draw_label_box(canvas_bgr, x1, y1, x2, y2, label)
            lines.append(f"{name}: {score:.2f}")

    # No usable detections — classify the full image and draw one frame box.
    if not lines:
        name, score = classify_crop(rgb)
        if score >= conf:
            margin = max(8, min(h, w) // 40)
            label = f"{name.upper()} {score:.2f}"
            draw_label_box(canvas_bgr, margin, margin, w - margin, h - margin, label)
            lines.append(f"{name}: {score:.2f}")

    return bgr_to_rgb(canvas_bgr), lines


def read_webcam_frame() -> np.ndarray | None:
    """Grab a single frame from the default camera via OpenCV."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None
    return bgr_to_rgb(frame_bgr)


def infer_image(rgb_image, conf: float):
    if rgb_image is None:
        return None, None, "No image provided."
    detected, lines = annotate_rgb(np.asarray(rgb_image), conf)
    return rgb_image, detected, format_results(lines)


def infer_video(video_path, conf: float):
    """Frame-by-frame OpenCV inference; show first input / processed frames."""
    if not video_path:
        return None, None, "No video provided."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "Could not open the video file."

    first_input = None
    first_detected = None
    collected: list[str] = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = bgr_to_rgb(frame_bgr)
        detected_rgb, frame_lines = annotate_rgb(frame_rgb, conf)
        collected.extend(frame_lines)

        if first_input is None:
            first_input = frame_rgb
            first_detected = detected_rgb

    cap.release()

    if first_input is None:
        return None, None, "The video contained no readable frames."

    unique_lines = list(dict.fromkeys(collected))
    return first_input, first_detected, format_results(unique_lines)


def toggle_source(mode: str):
    image_label = "Uploaded Image" if mode == "Image" else "Input Frame"
    detected_label = "Detected Image" if mode == "Image" else "Processed Frame"
    return (
        gr.update(visible=mode == "Image"),
        gr.update(visible=mode == "Video"),
        gr.update(visible=mode == "Webcam"),
        gr.update(label=image_label),
        gr.update(label=detected_label),
    )


def detect_objects(mode, image, video, webcam, conf):
    if mode == "Image":
        return infer_image(image, conf)
    if mode == "Video":
        return infer_video(video, conf)

    frame = webcam if webcam is not None else read_webcam_frame()
    if frame is None:
        return None, None, "No webcam frame captured."
    return infer_image(frame, conf)


with gr.Blocks(title="Waste Classification using YOLOv8") as demo:
    gr.Markdown("# Waste Classification using YOLOv8")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Image/Video Config")
            source = gr.Radio(
                choices=["Image", "Video", "Webcam"],
                value="Image",
                label="Source Selection",
            )
            image_in = gr.Image(
                label="Upload Image",
                type="numpy",
                sources=["upload"],
            )
            video_in = gr.Video(
                label="Upload Video",
                visible=False,
            )
            webcam_in = gr.Image(
                label="Webcam",
                type="numpy",
                sources=["webcam"],
                visible=False,
            )
            confidence = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.40,
                step=0.01,
                label="Confidence",
            )
            detect_btn = gr.Button("Detect Objects", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                input_view = gr.Image(label="Uploaded Image", interactive=False)
                output_view = gr.Image(label="Detected Image", interactive=False)
            with gr.Accordion("Detection Results", open=True):
                result_box = gr.Textbox(
                    label="Detected waste classes and confidence scores",
                    lines=8,
                    interactive=False,
                )

    source.change(
        fn=toggle_source,
        inputs=source,
        outputs=[image_in, video_in, webcam_in, input_view, output_view],
    )
    detect_btn.click(
        fn=detect_objects,
        inputs=[source, image_in, video_in, webcam_in, confidence],
        outputs=[input_view, output_view, result_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
