---
title: Waste Classification
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Waste Classification

This project compares two approaches for classifying household waste from a photo into six material types. The goal is to help recycling by naming the material, reporting how confident the model is, and treating weak scores as “not sure” rather than forcing a label.

Both notebooks use the same [TrashNet](https://github.com/garythung/trashnet) resized images.

| Notebook | Model | Role |
|----------|--------|------|
| [`waste-classification-with-cnn.ipynb`](waste-classification-with-cnn.ipynb) | Scratch CNN, then MobileNetV2 | Baseline + Keras transfer-learning experiment |
| [`waste-classification-with-yolov8.ipynb`](waste-classification-with-yolov8.ipynb) | YOLOv8n-cls | Stronger selected classifier |

**Selected system:** YOLOv8n-cls from `waste-classification-with-yolov8.ipynb` (**92.7%** validation top-1). MobileNetV2 is the better of the two CNN runs (**76.5%** val accuracy) but is clearly behind YOLO on this dataset.

## Problem and classes

Recycling only works if paper, plastic, glass, metal, and cardboard are separated. A photo classifier can suggest a bin, but it must also know when the image is too ambiguous.

| Class | Images | Share | What it looks like in this dataset |
|-------|--------|-------|------------------------------------|
| Paper | 594 | 23.5% | Sheets, newspaper, printed pages |
| Glass | 501 | 19.8% | Bottles and jars, often transparent |
| Plastic | 482 | 19.1% | Bottles, containers, film |
| Metal | 410 | 16.2% | Cans and tins |
| Cardboard | 403 | 15.9% | Boxes, corrugated board |
| Trash | 137 | 5.4% | Residual / mixed items that do not fit the others |

Total: **2,527** images. `trash` is about four times smaller than `paper`. That imbalance is the main reason leftover waste is harder to classify than a clean sheet of paper.

## Preprocessing

Both pipelines start from the same folder:

`C:\Users\LENOVO\Downloads\archive (1)\dataset-resized`

Shared choices:

- RGB images
- Resize to **224 × 224** (MobileNetV2 and YOLOv8n-cls both expect this scale)
- Pixel scaling to **\[0, 1\]** for the Keras models (`rescale=1./255`). YOLO applies its own letterbox / normalize pipeline internally.

### CNN notebook

Keras `ImageDataGenerator` with an 80/20 `validation_split` (**2,024** train / **503** val):

- rotation up to 30°
- width and height shift 0.2
- shear 0.15 and zoom 0.2
- horizontal flip

These transforms were chosen because TrashNet photos are mostly centered objects. Rotation, shift, and flip teach the network that a bottle on its side is still a bottle, without inventing unrealistic colors.

A later cell defined a *clean* validation generator (rescale only). That is the correct evaluation setup, but the logged MobileNet scores still come from the earlier generator, which also augmented validation images. That makes the 76.5% figure a slightly pessimistic estimate, not an inflated one.

### YOLOv8 notebook

Images are copied into a classification folder layout:

```
yolo_trashnet/train/<class>/*.jpg
yolo_trashnet/val/<class>/*.jpg
```

Each class is shuffled and split **80/20** (**2,019** train / **508** val). YOLO then applies its default classification augmentations (including `auto_augment=randaugment`, HSV jitter, and random erasing). No extra custom augmentations were added, because the defaults already cover lighting and mild distortion.

## Model architectures

### 1. From-scratch CNN (baseline only)

```
Conv2D(32) + ReLU + MaxPool
Conv2D(64) + ReLU + MaxPool
Conv2D(128) + ReLU + MaxPool
Flatten → Dense(256) + Dropout(0.5)
         → Dense(64)  + Dropout(0.5)
         → Dense(6)
```

This was a control: can a tiny network learn anything from 2.5k images? It cannot, at least not well. Flattening 224×224 feature maps also creates a huge dense layer and overfits quickly.

The output used **sigmoid + binary cross-entropy**. That is the wrong head for a single 6-class label. Softmax + categorical cross-entropy is the correct pairing. Train and validation also pointed at the **same folder**, so the validation number is leaked.

### 2. MobileNetV2 (CNN notebook, selected Keras model)

```
MobileNetV2 (ImageNet, include_top=False, 224×224×3)
→ GlobalAveragePooling2D
→ Dropout(0.3)
→ Dense(128, ReLU)
→ Dense(6, softmax)
```

Why this architecture:

- **Transfer learning** is the default for a small photo dataset. ImageNet already taught edges, textures, and bottle-like shapes.
- **MobileNetV2** is a lightweight inverted-residual network. It is small enough to train on CPU and still stronger than a scratch CNN.
- **Global average pooling** replaces Flatten, so the head has far fewer parameters and overfits less.
- **Dropout 0.3** and a small **128-unit** hidden layer keep the new head modest.
- **Softmax + categorical cross-entropy** matches a mutually exclusive 6-class problem.

Saved file: `waste_classifier_model.pkl`  
About **6.5M** parameters, **~2.0M** trainable after the later layers were unfrozen. Class order is alphabetical: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

### 3. YOLOv8n-cls (selected system)

```
YOLOv8n-cls (yolov8n-cls.pt)
→ classification head with 6 logits
→ softmax probabilities
```

Ultralytics reports **56 layers**, **1,445,974** parameters, **3.4 GFLOPs** (fused evaluate graph: 30 layers, 1.44M params).

Why this architecture:

- It is a modern classification backbone already pretrained on ImageNet-scale data.
- The **nano** variant is smaller than MobileNetV2 (~1.4M vs ~6.5M) and faster at inference (~6 ms/image on this CPU during val).
- Built-in training (AdamW, warmup, cosine-style schedule, RandAugment) is stronger than the hand-built Keras loop.
- Classification YOLO is the right tool here. Detection YOLO would be for bounding boxes; TrashNet is one object per image.

Best weights:

`C:\runs\classify\waste_classification\yolov8_trashnet\weights\best.pt`

The notebook run name is now `waste_classification_with_yolov8`. New training writes to that folder. The already-trained `best.pt` still loads from `yolov8_trashnet`.

## Training decisions

| Decision | CNN / MobileNetV2 | YOLOv8n-cls | Why |
|----------|-------------------|-------------|-----|
| Input size | 224×224 | 224×224 | Native size for both backbones; TrashNet images are already small |
| Split | 80/20 | 80/20 | Dataset is too small for a third test split and still have stable class counts |
| Batch size | 32 (final) | 32 | Fits CPU memory; 256 was used only on the weak baseline |
| Epochs | 15 + 15 | 25 | Enough to plateau without days of CPU training |
| Optimizer | Adam `1e-3` then `1e-5` | AdamW `lr=0.001` (Ultralytics auto) | Fine-tuning must use a small LR so ImageNet features are not destroyed |
| Loss | Categorical cross-entropy | Classification loss from Ultralytics | Multi-class, one label per image |
| Augmentation | Geometric Keras aug | YOLO defaults (RandAugment, HSV, erasing) | Reduce overfitting on 2k images |
| Class weights | Not used | Not used | Would be the next fix for the tiny `trash` class |

MobileNet training note: the first 15-epoch fit already had later MobileNet layers unfrozen at `1e-5`. There was no separate frozen-head warmup at `1e-3`. A cleaner recipe is: train the head only, then unfreeze. The run still improved, which is why it was kept, but that missing warmup is one reason it trails YOLO.

YOLOv8 trained for **25 epochs in 0.73 hours** on an 11th-gen i5 CPU (`torch-2.14.0+cpu`).

## Experiments and results

| Experiment | Notebook | Val metric | Verdict |
|------------|----------|------------|---------|
| Scratch CNN, 10 epochs | CNN | ~53% accuracy | Fail. Wrong loss, leaked val, too small a net |
| MobileNetV2 stage 1, 15 epochs | CNN | 66.4% accuracy | Learning, still weak |
| MobileNetV2 fine-tune, 15 more epochs | CNN | **76.5%** accuracy (best ~76.7%) | Usable Keras model, overfits (train 91%) |
| YOLOv8n-cls, 25 epochs | YOLO | **92.7% top-1** (99.8% top-5) | **Selected model** |

YOLOv8 validation top-1 by epoch (selected points):

| Epoch | Top-1 | Top-5 |
|-------|-------|-------|
| 1 | 73.0% | 98.4% |
| 2 | 83.5% | 99.8% |
| 8 | 90.7% | 100% |
| 12 | 92.3% | 99.8% |
| 20 | 92.9% | 99.8% |
| 25 (final val of `best.pt`) | **92.7%** | 99.8% |

Top-5 near 100% is expected with only six classes. The number that matters is **top-1**.

MobileNet train accuracy ended near **91%** while validation stayed near **76%**. That 15-point gap is overfitting. YOLO’s validation top-1 is much closer to a practical score, but it still has no held-out test set separate from the 20% val split.

## Reasoning behind specific predictions

Preprocessing at inference must match training: RGB, 224×224, Keras models divided by 255. YOLO accepts a path or NumPy image and does its own resize.

### `paper/paper100.jpg` — confident paper

The CNN notebook predicted **paper, 98.07%**. That is a typical TrashNet paper photo: a flat, light, fibrous sheet with print. Those textures are common in the 594 paper images, so both models should lock onto **paper**.

YOLOv8 is also right on clean paper, but Gradio logs show a more honest neighbor class:

- **paper 71%, cardboard 26%**

That split is reasonable. Paper and cardboard share color and fiber texture. A box flattened in frame, or a thick stack of paper, sits on that boundary. 71% vs 26% is still a clear paper call. If those two scores were 48% / 45%, the system should refuse.

### High-confidence trash

One Gradio example returned **trash 100%**. Residual-waste photos in TrashNet look messy and mixed (wrappers, food-soiled items). When the image matches that “none of the recyclable materials” look, both the label and the confidence are expected. The risk is the opposite error: a dirty recyclable item scored as trash because `trash` examples are dirty.

### Plastic vs trash — uncertain

- **plastic 62%, trash 33%**
- **plastic 42%, trash 23%, glass 19%**

These are the predictions that justify an abstain rule.

Plastic bottles can look like glass (transparency, specular highlights). Crushed plastic can look like trash. 42% is not a decision you should trust in a sorting line. A sensible rule:

- accept if top-1 ≥ **60%** and (top-1 − top-2) ≥ **15%**
- otherwise return: `Unable to confidently classify this image.`

Under that rule the 71% paper / 26% cardboard call is accepted, and the 42% plastic call is rejected.

### What a “cannot classify” case looks like

The models only know these six training folders. A shoe, a laptop, a dark blurry photo, or a pile of mixed items is out of distribution. Softmax will still pick a class and can look confident. That is why a threshold is required: low max probability, or two close scores, means **do not automate the bin choice**.

Example messages:

```
Prediction: Plastic
Confidence: 94.7%
```

```
Unable to confidently classify this image.
```

## Limitations and practical reliability

YOLOv8 is clearly better than MobileNetV2 on this benchmark, but **neither model is ready for unsupervised industrial sorting**.

- **Small data.** 2,527 studio-style images do not cover wet waste, bags, hands, or phone photos.
- **Class imbalance.** `trash` has 137 images. Errors will concentrate there and on plastic/glass/paper-cardboard pairs.
- **No true test set.** The 20% val split was used to pick `best.pt`. Reported 92.7% is slightly optimistic.
- **Domain shift.** TrashNet is clean, centered, single-object. Real bins are not.
- **Calibration.** A softmax 98% is not a 98% chance of being right, especially on unfamiliar photos.
- **CNN notebook issues.** The scratch model used the wrong loss and leaked validation data. MobileNet val images were augmented. Plots in that notebook still chart the scratch `hist`, not the MobileNet history.
- **App wiring.** `a.py` still loads the MobileNet pickle. The public demo and `app.py` use the stronger YOLOv8 weights.

Use YOLO as a suggestion engine: show the class and confidence, and hand the item to a person when confidence is low.

## Live demo

The Gradio app is deployed as a Hugging Face Space so it can be opened from any device:

https://huggingface.co/spaces/mariakawa/waste-classification

Upload an image there. If the model is not confident enough, it returns: `Unable to confidently classify this image.`

## How to run

### Train / inspect the CNN experiments

1. Point `waste-classification-with-cnn.ipynb` at the TrashNet `dataset-resized` folder.
2. Run cells in order. MobileNetV2 is the Keras model that gets pickled.
3. Requires `tensorflow`, `keras`, `numpy`, `opencv-python`, `matplotlib`, `pandas`, `tqdm`.

```bash
python a.py
```

Loads `waste_classifier_model.pkl` and opens a local Gradio UI.

### Train / use YOLOv8

1. Open `waste-classification-with-yolov8.ipynb`.
2. Run the split cell, then the `model.train(...)` cell (`epochs=25`, `imgsz=224`, `batch=32`).
3. Predict from `best.pt` or launch the notebook Gradio cell.
4. Requires `ultralytics`, `torch`, `matplotlib`, `pillow`, `gradio`.

Local YOLO app (same as the Space):

```bash
python app.py
```

## Project files

| File | Role |
|------|------|
| `waste-classification-with-cnn.ipynb` | Dataset EDA, scratch CNN, MobileNetV2 train, Keras prediction |
| `waste-classification-with-yolov8.ipynb` | YOLO split, YOLOv8n-cls train, prediction, Gradio demo |
| `waste_classifier_model.pkl` | Trained MobileNetV2 (Keras) |
| `weights/best.pt` | Trained YOLOv8n-cls weights |
| `app.py` | Hugging Face / Gradio app for YOLOv8 |
| `a.py` | Local Gradio app for the MobileNet model |
| `requirements.txt` | Dependencies for the Hugging Face Space |
