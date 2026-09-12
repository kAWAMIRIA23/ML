
import gradio as gr
import numpy as np
import cv2
import pickle
import tensorflow as tf

# 1. Load the model from the .pkl file
model_path = "waste_classifier_model.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    print("Successfully loaded model from PKL file!")
except Exception as e:
    print(f"Could not load PKL file ({e}). Using active model in memory instead.")

# 2. Define class names (Matches TrashNet order)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 3. Prediction Pipeline
def classify_waste(img):
    if img is None:
        return "No image uploaded."
    
    # Preprocess image to match training parameters (224x224, normalized 0-1)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Model inference
    predictions = model.predict(img_batch)[0]
    
    # Return dictionary of confidence scores for Gradio Label component
    confidence_scores = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    return confidence_scores

# 4. Build Gradio Interface
interface = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="numpy", label="Upload Waste Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="♻️ AI Waste Classifier",
    description="Upload an image of garbage (paper, glass, metal, plastic, cardboard, trash) to classify it using your trained model."
)

# Launch local Web UI
interface.launch(share=False)