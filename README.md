# Object Lens — 20-Class Object Detection & Deep Learning Pipeline

An end-to-end computer vision and deep learning project trained on the **Pascal VOC 2012** benchmark dataset (17,125 images, 40,138 object instances across 20 classes).

This repository combines **rigorous statistical machine learning and hypothesis testing** with modern **deep residual architectures** and **Faster R-CNN end-to-end object detection**, complete with an interactive **Flask** web application.

---

## 🌟 Key Highlights & Architecture

### 1. Data Pipeline & Exploratory Data Analysis (EDA)
- **Dataset**: Pascal VOC 2012 (20 object classes: *aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor*).
- **Automated XML Parsing**: Extracted and normalized bounding-box coordinates `[xmin, ymin, xmax, ymax]`, aspect ratios, and spatial metrics.
- **Feature Selection**: Evaluated dimensional features using **ReliefF**, **Mutual Information (Information Gain)**, and **Principal Component Analysis (PCA)**.

### 2. Deep Residual Neural Network (PyTorch)
- **Transfer Learning**: Extracted **2,048-dimensional feature representations** using a pre-trained **ResNet-50** backbone.
- **Custom Architecture (`DeepResidualNeuralNet`)**:
  - **Dense Projection**: `Linear(2048, 512)` $\to$ `BatchNorm1d` $\to$ `GELU` $\to$ `Dropout(0.3)`.
  - **Residual Skip Connection**: `Linear(512, 512)` $\to$ `BatchNorm1d` $\to$ `GELU` $\to$ `Dropout(0.3)` + Identity Shortcut ($y = \mathcal{F}(x) + x$) to eliminate vanishing gradients.
  - **Output Head**: `Linear(512, 20)` with label-smoothed Cross-Entropy loss.
  - **Optimizer**: AdamW with weight decay and Apple Silicon GPU acceleration (`mps`).

### 3. Rigorous 7-Stage Statistical Hypothesis Testing
To prove that the Deep Neural Network provides a statistically significant improvement over traditional machine learning baselines (Logistic Regression, Random Forest, Decision Tree), the notebook executes a formal statistical test battery:
- **Hypothesis Testing vs. Random Chance**: One-sample t-test ($H_0: \mu = 0.05$ vs. $H_1: \mu > 0.05$, $p < 0.001$).
- **10-Fold Stratified Cross-Validation**: Gathers fold score distributions across all models.
- **Paired t-Test**: Parametric test establishing statistically significant gains ($p < 0.05$).
- **Chi-Square Test**: Independence evaluation on the $20 \times 20$ prediction contingency matrix.
- **Friedman Test**: Non-parametric omnibus ranking across all models.
- **Wilcoxon Signed-Rank Test**: Paired non-parametric significance test.
- **One-Way ANOVA**: Variance analysis across model score distributions.
- **Nemenyi Post-Hoc Test**: Pairwise statistical significance matrix identifying top performers.

### 4. End-to-End Object Detection (Faster R-CNN ResNet-50-FPN-v2)
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN v2) extracting multi-scale feature maps.
- **Region Proposal Network (RPN)**: Generates candidate bounding box proposals directly from feature maps.
- **RoI Head**: Fast R-CNN RoIAlign predictor outputting both **exact bounding box coordinates** and **class confidence scores** in a single forward pass.

### 5. Web Deployment & Inference Engine
- **Flask Web Application**: Upload photos to view detected objects and confidence meters.
- **Multi-Scale Sliding Window**: Multi-scale crop generation (fractions: 0.8, 0.6, 0.4) with consensus voting to minimize false positives.

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
├── Object_Detection.ipynb       # Complete notebook: EDA, feature selection, DNN, hypothesis testing & Faster R-CNN
├── app.py                       # Flask web application server
├── detect.py                    # Inference script (multi-scale consensus voting engine)
├── saved_model/                 # Serialized model artifacts
│   ├── classifier.pkl           # Trained Deep Neural Network classifier
│   ├── scaler.pkl               # StandardScaler feature normalizer
│   └── label_encoder.pkl        # 20-class LabelEncoder
├── templates/
│   └── index.html               # Web UI template
├── static/
│   ├── style.css                # Modern responsive UI styles
│   └── app.js                   # Front-end preview & interaction logic
└── requirements.txt             # Project dependencies
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

```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser. Upload any JPG, PNG, or WebP photo to run detection.

### 3. Command Line Detection

Run detection directly on any image file:

```bash
python detect.py path/to/image.jpg --threshold 0.8 --min-votes 4
```

### 4. Running the Notebook

Open and run `Object_Detection.ipynb` in JupyterLab or VS Code:

```bash
jupyter lab Object_Detection.ipynb
```
> **Note**: Update `BASE_DIR` in Cell 4 to point to your local Pascal VOC 2012 directory if running the full data extraction and training from scratch.

---

## 🛠 Tech Stack

- **Deep Learning**: PyTorch, Torchvision (ResNet-50, Faster R-CNN FPN v2)
- **Statistical Testing**: SciPy (`scipy.stats`), Scikit-Posthocs (`posthoc_nemenyi_friedman`)
- **Machine Learning**: Scikit-Learn (Pipelines, StratifiedKFold, StandardScaler, Metrics)
- **Computer Vision & Image Processing**: OpenCV, Pillow (PIL)
- **Web & Serving**: Flask, Jinja2, HTML5/CSS3, JavaScript
