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


def resolve_weights() -> Path:
    for path in WEIGHTS_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No YOLOv8 weights found. Place best.pt (or best.onnx) under ./weights/"
    )


WEIGHTS = resolve_weights()
model = YOLO(str(WEIGHTS))


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def detection_lines(result, conf: float) -> list[str]:
    """Build human-readable class / confidence lines from one YOLO result."""
    lines: list[str] = []
    names = result.names

    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            score = float(box.conf[0])
            if score < conf:
                continue
            cls_id = int(box.cls[0])
            lines.append(f"{names[cls_id]}: {score:.2f}")
        return lines

    # Classification head (YOLOv8-cls): report ranked class probabilities.
    if result.probs is not None:
        scores = result.probs.data.cpu().numpy()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        for cls_id, score in ranked:
            if float(score) < conf:
                continue
            lines.append(f"{names[int(cls_id)]}: {float(score):.2f}")
    return lines


def format_results(all_lines: list[str]) -> str:
    if not all_lines:
        return "No waste objects detected above the confidence threshold."
    return "\n".join(all_lines)


def overlay_classify_box(rgb_image: np.ndarray, result, conf: float) -> np.ndarray:
    """Draw a labeled frame when the model is classification-only (no boxes)."""
    annotated = rgb_image.copy()
    if result.probs is None:
        return annotated

    score = float(result.probs.top1conf)
    if score < conf:
        return annotated

    name = result.names[int(result.probs.top1)]
    height, width = annotated.shape[:2]
    color = (31, 122, 70)
    cv2.rectangle(annotated, (10, 10), (width - 10, height - 10), color, 3)
    label = f"{name} {score:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(annotated, (18, 18), (28 + text_w, 32 + text_h), color, -1)
    cv2.putText(
        annotated,
        label,
        (24, 24 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def annotate_rgb(rgb_image: np.ndarray, conf: float) -> tuple[np.ndarray, list[str]]:
    """Run YOLO predict on an RGB frame and return an RGB annotated image."""
    results = model.predict(rgb_image, conf=conf, verbose=False)
    result = results[0]

    # result.plot() returns BGR; convert back to RGB for Gradio.
    plotted_bgr = result.plot()
    plotted_rgb = bgr_to_rgb(plotted_bgr)

    if result.boxes is None or len(result.boxes) == 0:
        plotted_rgb = overlay_classify_box(plotted_rgb, result, conf)

    return plotted_rgb, detection_lines(result, conf)


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
        detected_rgb, lines = annotate_rgb(frame_rgb, conf)
        collected.extend(lines)

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
        # Left: config / source panel
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

        # Right: results panel
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
