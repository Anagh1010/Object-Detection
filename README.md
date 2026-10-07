# Object Lens — 20-Class Object Detection & Deep Learning Pipeline

An end-to-end computer vision and machine learning project trained on the **Pascal VOC 2012** benchmark dataset (17,125 images, 40,138 object instances across 20 classes).

This repository combines a **14-section Exploratory Data Analysis**, **multi-method feature selection**, **ResNet-50 CNN embeddings**, **three classical ML classifiers** compared with rigorous **statistical hypothesis testing**, and a **Faster R-CNN end-to-end detection** pipeline — served through an interactive **Flask** web application.

---

## 🌟 Pipeline Overview

### 1. Dataset & Exploratory Data Analysis (14 Sections)
- **Dataset**: Pascal VOC 2012 — 20 object classes: *aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor*.
- **XML Annotation Parsing**: Extracted and normalised bounding-box coordinates `[xmin, ymin, xmax, ymax]`, image dimensions, aspect ratios, spatial metrics, object flags (`difficult`, `truncated`, `occluded`), and pose labels.
- **EDA Sections**:
  1. Dataset Structure & Paths
  2. Parse Annotations into DataFrame
  3. Missing Values & Data Integrity
  4. Class Distribution & Imbalance Analysis
  5. Objects-per-Image Distribution
  6. Image Size & Aspect Ratio Analysis
  7. Bounding Box Analysis & Outlier Detection (IQR)
  8. Difficult / Truncated / Occluded Object Analysis
  9. Train / Validation Split Analysis
  10. Segmentation Mask Analysis
  11. Sample Image Visualisations with Bounding Boxes
  12. Correlation & Class Co-occurrence Analysis
  13. Colour / Brightness Statistics
  14. Summary & Recommendations

### 2. Feature Extraction & Selection

**12 numeric features** per object instance used across all stages:

| Feature | Description |
| :--- | :--- |
| `img_w`, `img_h`, `img_area` | Image dimensions and area |
| `bb_w`, `bb_h`, `bb_area` | Bounding box width, height, area |
| `bb_rel_area` | Bounding box area relative to image area |
| `aspect_ratio` | Image width / height |
| `bb_aspect` | Bounding box width / height |
| `difficult`, `truncated`, `occluded` | Object annotation flags |

**Feature selection methods compared side-by-side:**
- **Pearson Correlation** — linear relationship with encoded label
- **ReliefF** — distance-based, class-boundary sensitivity
- **Mutual Information (Information Gain)** — non-linear dependency via `mutual_info_classif`
- **PCA** — principal component loadings (variance decomposition, 95% threshold)

### 3. CNN Feature Embeddings (ResNet-50)

- Pretrained **ResNet-50** backbone (`ResNet50_Weights.DEFAULT`) with the final classification head replaced by `nn.Identity()` to extract **2,048-dimensional embeddings** per object crop.
- Object crops extracted per bounding box from `JPEGImages/` using the annotation DataFrame.
- **Balanced subsample**: up to 500 instances per class (10,000 total cap) to make extraction tractable.
- Device: Apple Silicon MPS (`mps`) if available, else CPU.

### 4. Classical ML Model Training

Three classifiers trained on the ResNet-50 embeddings with 80/20 stratified split:

| Model | Notes |
| :--- | :--- |
| **Logistic Regression** | `lbfgs` solver, `max_iter=500`, `class_weight='balanced'` |
| **Random Forest** | `n_estimators=100`, `class_weight='balanced'` |
| **Decision Tree** | `class_weight='balanced'` |

Evaluation: Accuracy, Weighted F1, Macro F1, Weighted Precision, Weighted Recall, Confusion Matrix, multi-class ROC-AUC.

### 5. Rigorous 6-Stage Statistical Hypothesis Testing

10-fold Stratified Cross-Validation gathers fold-level F1 scores across all three models, which feed into a full statistical test battery:

| Test | Purpose |
| :--- | :--- |
| **One-sample t-test** | Model accuracy vs. random-chance baseline ($H_0: \mu = 0.05$) |
| **Paired t-Test** | Parametric pairwise comparison (LR vs RF) |
| **Chi-Square Test** | Independence of predicted vs. actual class distributions |
| **Friedman Test** | Non-parametric omnibus ranking across all 3 models |
| **Wilcoxon Signed-Rank Test** | Paired non-parametric significance (LR vs RF) |
| **ANOVA** | Variance analysis across model score distributions |
| **Nemenyi Post-Hoc Test** | Pairwise p-value matrix (via `scikit-posthocs`) |

### 6. End-to-End Object Detection (Faster R-CNN ResNet-50-FPN-v2)

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN v2) for multi-scale feature maps.
- **Region Proposal Network (RPN)**: Proposes candidate bounding boxes directly from feature maps.
- **RoI Head**: `FastRCNNPredictor` with RoIAlign pooling — outputs exact bounding box coordinates and class confidence scores in a single forward pass.
- **21 classes**: 20 VOC object classes + background (index 0).
- Custom `VOCDetectionDataset` PyTorch `Dataset` with horizontal-flip augmentation and support for `train.txt` / `val.txt` ImageSets splits.
- Trained with SGD (`lr=0.005`, momentum=0.9, weight decay=5e-4) + `StepLR` scheduler.

### 7. Web Deployment & Inference Engine

- **Flask Web Application** (`app.py`): Upload JPG, PNG, or WebP photos to view detected objects and confidence meters.
- **Inference** (`detect.py`): Multi-scale sliding-window crop generation (scale fractions: 0.8, 0.6, 0.4, stride 30%) with consensus voting to minimise false positives. Confidence threshold: 0.8, minimum votes: 4.

---

## 📊 Performance Summary

| Metric | Result |
| :--- | :--- |
| **Test Accuracy** | **87.25%** |
| **Weighted F1-Score** | **87.22%** |
| **Macro ROC-AUC** | **0.993** |
| **Statistical Significance** | **$p < 0.05$** across Paired t-test, ANOVA, Friedman, and Wilcoxon tests |

---

## 📁 Repository Structure

```
├── Object_Detection.ipynb       # Main notebook: EDA → Feature Selection → CNN Embeddings →
│                                #   Model Training → Hypothesis Testing → Faster R-CNN
├── app.py                       # Flask web application server
├── detect.py                    # Inference script (multi-scale consensus voting engine)
├── saved_model/                 # Serialised model artifacts
│   ├── classifier.pkl           # Trained classifier (LogisticRegression on ResNet-50 embeddings)
│   ├── scaler.pkl               # StandardScaler feature normaliser
│   └── label_encoder.pkl        # 20-class LabelEncoder
├── templates/
│   └── index.html               # Web UI template
├── static/
│   ├── style.css                # Responsive UI styles
│   └── app.js                   # Front-end preview & interaction logic
├── requirements-original.txt    # Core ML/notebook dependencies
└── extra-requirements-for-app.txt  # Flask & web-app dependencies
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Anagh1010/Object-Detection.git
cd Object-Detection

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-original.txt
pip install -r extra-requirements-for-app.txt
```

### 2. Run the Web Application

The `saved_model/` artifacts are already included — no need to re-run the notebook to use the app.

```bash
python app.py
```

Open **`http://127.0.0.1:5000`** in your browser. Upload any JPG, PNG, or WebP photo to run detection.

### 3. Command Line Detection

```bash
python detect.py path/to/image.jpg --threshold 0.8 --min-votes 4
```

### 4. Running the Notebook

Open and run `Object_Detection.ipynb` in JupyterLab or VS Code:

```bash
jupyter lab Object_Detection.ipynb
```

> **Dataset path**: Update `BASE_DIR` in **Cell 4** to point to your Pascal VOC 2012 directory.  
> Expected structure: `BASE_DIR/JPEGImages/`, `BASE_DIR/Annotations/`, `BASE_DIR/ImageSets/Main/`, `BASE_DIR/SegmentationClass/`.  
> If your dataset is on an external SSD, set `BASE_DIR` to the full mount path (e.g. `/Volumes/<SSD-name>/...`).

---

## 🛠 Tech Stack

- **Deep Learning**: PyTorch, Torchvision (ResNet-50, Faster R-CNN FPN v2)
- **Feature Selection**: scikit-learn (`mutual_info_classif`, PCA), skrebate (`ReliefF`)
- **Statistical Testing**: SciPy (`scipy.stats`), scikit-posthocs (`posthoc_nemenyi_friedman`)
- **Machine Learning**: scikit-learn (Logistic Regression, Random Forest, Decision Tree, Pipelines, StratifiedKFold, StandardScaler, Metrics)
- **Computer Vision & Image Processing**: Pillow (PIL)
- **Data**: pandas, NumPy, Matplotlib, Seaborn, tqdm
- **Web & Serving**: Flask, Jinja2, HTML5/CSS3, JavaScript
