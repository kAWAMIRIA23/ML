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

This project decides **how household-waste photos should be explored and prepared**, **which modeling approaches fit the data**, **how those models should be trained**, and **how to judge whether the deployed solution is reliable**.

The practical goal is to suggest a recycling bin (paper, plastic, glass, metal, cardboard, or residual trash), report confidence, and abstain when the score is too weak to trust.

| Notebook | Model | Role |
|----------|--------|------|
| [`waste-classification-with-cnn.ipynb`](waste-classification-with-cnn.ipynb) | Scratch CNN, then MobileNetV2 | Baseline + Keras transfer-learning experiment |
| [`waste-classification-with-yolov8.ipynb`](waste-classification-with-yolov8.ipynb) | YOLOv8n-cls | Selected classifier |

**Selected system:** YOLOv8n-cls (**92.7%** validation top-1). MobileNetV2 reached **76.5%** val accuracy and is kept as the Keras baseline. Both notebooks use [TrashNet](https://github.com/garythung/trashnet) resized images.

---

## 1. Exploring and preparing the images

### What to look at first (EDA)

Before training, inspect the folder of class-labeled photos (`dataset-resized`) the same way the CNN notebook does:

1. **Counts per class** — is the problem balanced?
2. **A few samples per class** — lighting, background, single vs mixed objects, blur, occlusion.
3. **Failure modes you can already see by eye** — paper vs cardboard, clear plastic vs glass, dirty recyclables vs `trash`.
4. **Domain fit** — TrashNet is mostly clean, centered, single-object studio photos. Phone photos of bins, piles, or people in frame are out of distribution and need different prep (and often detection labels) if that is the product goal.

| Class | Images | Share | Typical look in TrashNet |
|-------|--------|-------|--------------------------|
| Paper | 594 | 23.5% | Sheets, newspaper, printed pages |
| Glass | 501 | 19.8% | Bottles and jars, often transparent |
| Plastic | 482 | 19.1% | Bottles, containers, film |
| Metal | 410 | 16.2% | Cans and tins |
| Cardboard | 403 | 15.9% | Boxes, corrugated board |
| Trash | 137 | 5.4% | Residual / mixed items |

Total: **2,527** images. `trash` is about four times smaller than `paper`, so leftover waste is harder to learn and evaluate fairly.

### How images should be prepared

Shared prep for this project:

| Step | Choice | Why |
|------|--------|-----|
| Color | RGB | Matches how both Keras and Ultralytics load photos |
| Size | **224 × 224** | Native scale for MobileNetV2 and YOLOv8n-cls; TrashNet images are already small |
| Keras scaling | divide by 255 → \[0, 1\] | Matches `ImageDataGenerator(rescale=1./255)` |
| YOLO scaling | leave to Ultralytics | Letterbox + normalize happen inside `predict` / `train` |
| Split | **80% train / 20% val**, shuffled per class | Enough images per class for a stable validation count on 2.5k photos |
| Leakage check | train and val must not share the same files | The scratch CNN pointed train and val at the same folder — that invalidates its score |

**Augmentation (train only):**

- CNN notebook: rotation ≤ 30°, width/height shift 0.2, shear 0.15, zoom 0.2, horizontal flip — reasonable for centered objects without inventing unrealistic colors.
- YOLOv8-cls: Ultralytics defaults (RandAugment, HSV jitter, random erasing).

**Validation images should not be heavily augmented** when you report the final number. The MobileNet score still came from a generator that augmented val; treat **76.5%** as slightly pessimistic, not optimistic.

**YOLO classification layout used for training:**

```
yolo_trashnet/train/<class>/*.jpg
yolo_trashnet/val/<class>/*.jpg
```

**If the product needs bounding boxes on outdoor piles**, preparation changes: you need detect-style labels (`class cx cy w h`), hard-negative scenes (people, banners, streets with empty `.txt` files), and usually a higher `imgsz`. TrashNet alone (one label per image, no boxes) is the right prep for **classification**, not for industrial multi-object detection.

Source folder used in the notebooks:

`C:\Users\LENOVO\Downloads\archive (1)\dataset-resized`

---

## 2. Which modeling approaches are appropriate

Match the model to the **label type** and **scene type**.

| Scenario | Appropriate approach | Inappropriate approach |
|----------|----------------------|-------------------------|
| One waste item, whole-image class (TrashNet) | Image **classification** (YOLOv8-cls, MobileNetV2, etc.) | Detection trained without box labels |
| Many items in one photo, need boxes | Object **detection** (YOLOv8 detect) with labeled boxes + negatives | Forcing a classifier onto every crop of people/background |
| Tiny custom dataset | Transfer learning from ImageNet / YOLO pretrained weights | Training a large net from scratch |
| Need “I don’t know” behavior | Softmax + **confidence / margin thresholds** | Always returning the argmax class |

### Approaches compared in this repo

**1. From-scratch CNN (control only)**  
Small ConvNet + dense head. Useful as a baseline, not as a product model: too few inductive biases, easy to overfit, and the notebook run used the wrong loss (sigmoid + BCE for a single 6-way label) plus leaked validation.

**2. MobileNetV2 (transfer learning, Keras)**  
Good default when data is small: freeze or gently fine-tune an ImageNet backbone, GAP instead of Flatten, softmax + categorical cross-entropy. Competitive but clearly behind YOLO here (**76.5%** val).

**3. YOLOv8n-cls (selected)**  
Classification YOLO pretrained at scale, nano size (~1.4M params), strong built-in training recipe. Best fit for TrashNet’s single-object labels (**92.7%** top-1). Detection YOLO is for boxes; it is the wrong primary tool until you have detection labels.

**App note:** `app.py` can also propose regions with a general detector and classify crops with `weights/best.pt`. That is a demo path for boxes on simple photos. It is **not** a substitute for a detector trained on your outdoor waste + negatives, and it can mislabel people or banners if used carelessly.

---

## 3. How the models should be trained

### Training principles used here

1. **Start from pretrained weights** (ImageNet / Ultralytics).
2. **Fine-tune with a modest learning rate** so useful low-level features are not wiped out.
3. **Augment train only**; keep val as a clean probe when possible.
4. **Train long enough to plateau**, then stop (early stopping / best checkpoint).
5. **Save `best.pt` from validation**, not from the last epoch alone.
6. **Do not tune on the same split you report as final truth** without a separate holdout (see evaluation).

### Settings that were used

| Decision | CNN / MobileNetV2 | YOLOv8n-cls | Why |
|----------|-------------------|-------------|-----|
| Input size | 224×224 | 224×224 | Native for both; images already small |
| Split | 80/20 | 80/20 | Small data; keep class counts usable |
| Batch size | 32 | 32 | Fits CPU memory |
| Epochs | 15 + 15 fine-tune | 25 | Enough to plateau on this set |
| Optimizer | Adam `1e-3` then `1e-5` | AdamW `lr≈0.001` | Lower LR when unfreezing backbone |
| Loss | Categorical CE | Ultralytics cls loss | One mutually exclusive label |
| Augmentation | Geometric Keras aug | YOLO defaults | Fight overfitting on ~2k images |
| Class weights | Not used | Not used | Next fix for scarce `trash` |

YOLOv8 training took about **0.73 hours** for 25 epochs on an 11th-gen i5 CPU.

Best classification weights in this repo: `weights/best.pt`.

### How to retrain

**CNN / MobileNet**

1. Point `waste-classification-with-cnn.ipynb` at `dataset-resized`.
2. Run EDA → scratch baseline (optional) → MobileNetV2.
3. Prefer: train head first, then unfreeze later layers at a low LR.

**YOLOv8-cls (recommended)**

1. Open `waste-classification-with-yolov8.ipynb`.
2. Build `yolo_trashnet` train/val folders.
3. `model.train(..., epochs=25, imgsz=224, batch=32)`.
4. Deploy `best.pt` into `weights/best.pt` and run `python app.py`.

---

## 4. How I evaluate whether the solution is reliable

Reliability is not a single accuracy number. I treat it as a checklist.

### A. Offline metrics on a clean validation split

| Check | What I did / expect | Pass signal |
|-------|---------------------|-------------|
| Top-1 accuracy | Primary metric for 6-way classification | YOLO **92.7%** vs MobileNet **76.5%** → YOLO selected |
| Top-5 | Almost always high with only 6 classes | Informative but not decisive |
| Train vs val gap | MobileNet train ~91% / val ~76% | Large gap → overfitting risk |
| Per-class behavior | Watch `trash`, plastic/glass, paper/cardboard | Weak minority class or confusions → not fully reliable |
| Correct protocol | No train/val file overlap; val not over-augmented | Scratch CNN failed this; its ~53% is not trustworthy |

Selected YOLO epoch trace (validation top-1):

| Epoch | Top-1 | Top-5 |
|-------|-------|-------|
| 1 | 73.0% | 98.4% |
| 2 | 83.5% | 99.8% |
| 8 | 90.7% | 100% |
| 12 | 92.3% | 99.8% |
| 20 | 92.9% | 99.8% |
| 25 (`best.pt`) | **92.7%** | 99.8% |

### B. Decision quality, not just argmax

Softmax always returns a class. A reliable system must sometimes say **no**.

Rule used in analysis:

- accept if top-1 ≥ **60%** and (top-1 − top-2) ≥ **15%**
- otherwise: `Unable to confidently classify this image.`

Examples that justify that rule:

- `paper 71%` / `cardboard 26%` → accept paper (clear margin).
- `plastic 42%` / `trash 23%` / `glass 19%` → reject; too close to automate a bin.

### C. Qualitative / error review

I spot-check real files (for example `paper/paper100.jpg`) and Gradio outputs:

- Does a confident call look visually obvious?
- Are near-misses the expected confusions (paper↔cardboard, plastic↔glass)?
- Do out-of-distribution photos (shoes, laptops, dark blur, mixed piles) get wrongly high confidence?

If yes to the last point, the solution is **not** reliable for unsupervised use without an abstain rule or more data.

### D. Domain-shift and product tests

TrashNet val accuracy **overstates** real-world reliability when users upload:

- outdoor piles, bags, hands, wet waste, or phone photos;
- scenes with people, signs, or banners (especially if a crop-classifier is applied to every detected region).

A release checklist I use:

1. Hold out a small **phone-photo / bin** set that never entered training.
2. Measure accuracy **and** abstain rate under the confidence rule.
3. Prefer missing a call over a confident wrong bin (cost of mis-sort).
4. Confirm preprocessing at inference matches training (RGB; Keras ÷255; YOLO via Ultralytics).
5. Compare models on the **same** split before selecting one for `app.py`.

### E. What “reliable enough” means for this project

| Use case | Reliable? |
|----------|-----------|
| Demo / education / assisted sorting with a human in the loop | **Yes**, with YOLO + confidence shown |
| Fully automatic industrial sorting on TrashNet alone | **No** |
| Outdoor multi-object detection without box-labeled data + negatives | **No** |

**Bottom line:** YOLOv8n-cls is the selected model because it clearly wins the controlled TrashNet comparison, but reliability in production still depends on abstain thresholds, imbalance handling, and testing on photos that look like the real deployment domain—not only on studio TrashNet validation.

---

## Limitations (summary)

- Small, studio-style dataset (~2.5k images).
- Class imbalance (`trash` is scarce).
- No separate test set beyond the 20% val used to pick `best.pt` → 92.7% is slightly optimistic.
- Softmax confidence is not well calibrated out of distribution.
- `a.py` still demos MobileNet; `app.py` and the public Space use YOLOv8 `weights/best.pt`.

---

## Live demo and how to run

Gradio on Hugging Face (embedded from Netlify via `netlify-demo/`):

https://huggingface.co/spaces/mariakawa/waste

```bash
# Local YOLO app
python app.py

# MobileNet demo
python a.py

# Netlify folder preview
python -m http.server 8080 --directory netlify-demo
```

Weights / Space files: https://huggingface.co/mariakawa/waste-classification

## Project files

| File | Role |
|------|------|
| `waste-classification-with-cnn.ipynb` | EDA, scratch CNN, MobileNetV2 |
| `waste-classification-with-yolov8.ipynb` | YOLO split, YOLOv8n-cls train, Gradio |
| `waste_classifier_model.pkl` | Trained MobileNetV2 |
| `weights/best.pt` | Trained YOLOv8n-cls |
| `app.py` | Gradio YOLOv8 app (image / video / webcam) |
| `a.py` | Gradio app for MobileNet |
| `requirements.txt` | Space dependencies |
| `netlify-demo/` | Static page embedding the HF Space |
| `netlify.toml` | Publishes `netlify-demo` |
