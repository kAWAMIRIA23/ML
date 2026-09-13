import os
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt"
model = YOLO(str(WEIGHTS))

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15


def classify_waste(img):
    if img is None:
        return "No image uploaded."

    results = model(img)
    probs = results[0].probs
    names = results[0].names
    scores = {names[i]: float(probs.data[i]) for i in range(len(names))}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_conf = ranked[0]
    second_conf = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_conf < CONFIDENCE_THRESHOLD or (top_conf - second_conf) < MARGIN_THRESHOLD:
        return "Unable to confidently classify this image."

    return scores


demo = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="numpy", label="Upload waste image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Waste Classification with YOLOv8",
    description=(
        "Upload a photo of cardboard, glass, metal, paper, plastic, or trash. "
        "The model returns the class and confidence. If the top score is below 60% "
        "or the top two classes are too close, it will not force a label."
    ),
    examples=None,
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
