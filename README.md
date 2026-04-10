# PatchCore VisA – Anomaly Detection Pipeline

> **Train PatchCore variants on the VisA dataset in Kaggle, export artifacts, and run a Windows Streamlit GUI for interactive anomaly inspection with pixel-thresholded heatmaps and calibrated confidence scores.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Part 1 – Kaggle Training & Export](#part-1--kaggle-training--export)
4. [Part 2 – Windows Streamlit GUI](#part-2--windows-streamlit-gui)
5. [Artifact Format Reference](#artifact-format-reference)
6. [Model Variants](#model-variants)
7. [Metrics & Thresholds](#metrics--thresholds)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements four PatchCore anomaly-detection variants evaluated on all **12 VisA object categories**:

| Variant | Key | Description |
|---------|-----|-------------|
| Standard PatchCore | `standard` | k-NN distance in WideResNet-50-2 feature space |
| FR-PatchCore | `fr` | PCA feature reconstruction error + k-NN distance |
| Masked PatchCore | `mask` | Foreground-masked memory bank (Otsu thresholding) |
| FR+Masked PatchCore | `fr_mask` | Combined FR and Masking |

**Feature extractor**: WideResNet-50-2 (ImageNet pretrained), layers `layer2` + `layer3`  
**Metrics**: Image AUROC/AUPRC, Pixel AUROC/AUPRC, per-category thresholds (best-F1 and FPR-target)  
**Confidence**: Calibrated probability P(anomaly) via Isotonic Regression  
**GUI**: Streamlit app with pixel-thresholded heatmaps showing only anomalous regions

---

## Repository Structure

```
visa/
├── README.md
├── requirements.txt          ← Training dependencies (Kaggle / GPU machine)
├── requirements_ui.txt       ← UI dependencies (Windows, CPU-OK)
├── .gitignore
│
├── src/                      ← Shared Python modules
│   ├── __init__.py
│   ├── dataset.py            ← VisADataset, dataloaders, path discovery
│   ├── patchcore.py          ← All 4 model variants
│   ├── metrics.py            ← AUROC, AUPRC, threshold selection
│   ├── calibration.py        ← Probability calibration (isotonic)
│   ├── visualization.py      ← Heatmap generation & overlays
│   └── artifacts.py          ← Export / import utilities
│
├── kaggle/                   ← Training pipeline (run in Kaggle)
│   ├── config.yaml           ← All hyperparameters
│   ├── train_kaggle.py       ← Main CLI training script
│   └── patchcore_visa.ipynb  ← Ready-to-run Kaggle notebook
│
└── ui/                       ← Streamlit GUI (run on Windows)
    └── app.py
```

---

## Part 1 – Kaggle Training & Export

### Step 1.1 – Set up the Kaggle notebook

1. Go to [Kaggle](https://www.kaggle.com) → **Notebooks** → **New Notebook**
2. Click **File** → **Import Notebook** and upload `kaggle/patchcore_visa.ipynb`  
   *(or create a new notebook and paste `kaggle/train_kaggle.py` as a script)*
3. In the notebook, click **+ Add Data** → search for **"VisA"** and add the dataset  
   *(the dataset will be mounted at `/kaggle/input/visa/` or similar)*
4. Enable **GPU accelerator**: Settings → Accelerator → **GPU T4 x2** (or P100)
5. Set internet access: Settings → Internet → **On** (needed for pretrained weights)

### Step 1.2 – Clone the repository inside Kaggle

In the first code cell of your notebook:
```python
!git clone https://github.com/Annmary55/visa.git /kaggle/working/visa
import sys
sys.path.insert(0, '/kaggle/working/visa')
```

### Step 1.3 – Install extra dependencies

```python
!pip install -q scikit-image scikit-learn tqdm pyyaml joblib
```

### Step 1.4 – Run training

#### Option A: Use the notebook (`patchcore_visa.ipynb`)
Just run all cells in order. The notebook will:
- Detect the VisA dataset location automatically
- Train all 4 variants on all 12 categories
- Evaluate on val and test splits
- Compute thresholds and metrics
- Fit confidence calibrators
- Save a downloadable ZIP file

#### Option B: Use the CLI script
```bash
python /kaggle/working/visa/kaggle/train_kaggle.py \
    --config /kaggle/working/visa/kaggle/config.yaml \
    --output-dir /kaggle/working/exports \
    --variants standard fr mask fr_mask \
    --categories all
```

To run a quick smoke test on just one category:
```bash
python /kaggle/working/visa/kaggle/train_kaggle.py \
    --config /kaggle/working/visa/kaggle/config.yaml \
    --output-dir /kaggle/working/exports \
    --dry-run
```

### Step 1.5 – Download the exports

After training completes:
1. In the Kaggle notebook output panel, find `patchcore_visa_exports.zip`
2. Click the three-dot menu → **Download**
3. Save it to your Windows machine

The ZIP contains:
```
exports/
├── standard/
│   ├── thresholds.json
│   ├── metrics_summary.json
│   ├── calibrators/
│   │   ├── candle.pkl
│   │   ├── capsules.pkl
│   │   └── ...
│   └── candle/
│       ├── memory_bank.npz
│       └── model_config.json
├── fr/
│   └── ...
├── mask/
│   └── ...
├── fr_mask/
│   └── ...
├── categories.json
└── all_metrics_summary.json
```

---

## Part 2 – Windows Streamlit GUI

### Step 2.1 – Install Python (Windows)

Download Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/windows/).  
During installation, check **"Add Python to PATH"**.

### Step 2.2 – Clone the repository

Open **PowerShell** or **Command Prompt**:
```powershell
git clone https://github.com/Annmary55/visa.git
cd visa
```

### Step 2.3 – Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 2.4 – Install UI dependencies

```powershell
pip install -r requirements_ui.txt
```

> **Note**: For CPU-only Windows (no NVIDIA GPU), PyTorch defaults to CPU. Inference is still fast enough for single-image testing.

### Step 2.5 – Prepare artifacts and test data

**Extract the Kaggle ZIP:**
```powershell
# Create an exports directory and extract
mkdir exports
# Extract patchcore_visa_exports.zip into the exports/ folder
# You can do this with Windows Explorer (right-click → Extract All)
# or via PowerShell:
Expand-Archive patchcore_visa_exports.zip -DestinationPath exports
```

**Download and extract the VisA test images:**

From the Kaggle dataset page, download the VisA dataset and extract the **test** split:
```
data/
└── visa/
    └── test/
        ├── candle/
        │   ├── good/          ← normal test images
        │   └── bad/           ← anomaly test images
        ├── capsules/
        └── ...
```

> Alternatively, the training script can extract test images from the full VisA dataset.  
> VisA categories: `candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum`

### Step 2.6 – Run the Streamlit GUI

```powershell
# From the repo root
streamlit run ui/app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

### Step 2.7 – Using the GUI

#### Sidebar controls
| Control | Description |
|---------|-------------|
| **Model Variant** | Standard or FR (default); check "Show Advanced Models" to also see Mask and FR+Mask |
| **Category** | Select one of the 12 VisA categories |
| **Threshold Mode** | "Best F1" (maximizes F1 on validation) or "Low False Alarm" (FPR ≤ 1%) |
| **Artifacts directory** | Path to your extracted `exports/` folder |
| **Test data directory** | Path to your `data/visa/test/` folder |

#### Tabs
| Tab | Description |
|-----|-------------|
| **Browse Test Set** | Browse and analyze images from your local test folder |
| **Upload Image** | Upload any JPG/PNG for single-image inspection |
| **Batch Analysis** | Run inference on the full test set, see metrics and score distributions |
| **Metrics Dashboard** | View per-category metrics table and AUROC/AUPRC bar charts |

#### Reading the results
- **Predicted label**: ANOMALY (red) or NORMAL (green)
- **Anomaly score**: Raw PatchCore score (higher = more anomalous)
- **Threshold**: The selected threshold for this category
- **Calibrated Confidence**: Isotonic-calibrated P(anomaly) and P(normal), 0–100%
- **Heatmap overlay**: Only pixels **above the pixel threshold** are colored, highlighting anomalous regions

---

## Artifact Format Reference

### `thresholds.json`
```json
{
  "variant": "standard",
  "fpr_target": 0.01,
  "categories": {
    "candle": {
      "thr_img_f1":  3.21,
      "thr_img_fpr": 4.05,
      "thr_px_f1":   0.87,
      "thr_px_fpr":  0.93,
      "img_auroc":   0.982,
      "img_auprc":   0.975,
      "pixel_auroc": 0.961,
      "pixel_auprc": 0.734,
      "img_f1":      0.921,
      "img_precision": 0.904,
      "img_recall":  0.939
    }
  }
}
```

### Memory bank (`memory_bank.npz`)
```
memory_bank.npz:
  patches   – float32 array (N, C)
  pca_components – (fr/fr_mask only) PCA projection matrix
```

### Calibrators (`calibrators/<category>.pkl`)
Serialized `ScoreCalibrator` object (joblib). Call `.predict_proba([score])` to get P(anomaly).

---

## Model Variants

### Standard PatchCore
Uses WideResNet-50-2 features from `layer2` (stride 8) and `layer3` (stride 16).  
Layer3 is upsampled to match layer2, features are concatenated (512+1024=1536-d), and a 3×3 average pool provides neighborhood context.  
A greedy k-center coreset (10% of patches) compresses the memory bank.  
At inference: k=5 nearest neighbors → max over patches = image score.

### FR-PatchCore
Same feature extraction, then PCA (256 components) is fit on the memory bank.  
Anomaly score = 0.5 × k-NN distance + 0.5 × PCA reconstruction error.  
This tends to produce smoother anomaly maps.

### Masked PatchCore
Otsu thresholding on the grayscale training image generates a foreground mask.  
Only patches within the foreground enter the memory bank (reduces background noise in scoring).  
At inference: anomaly map is zeroed outside the foreground, reducing false alarms on backgrounds.

### FR+Masked PatchCore
Combines both: foreground-filtered memory bank + PCA reconstruction scoring.  
Best overall for objects with cluttered backgrounds.

---

## Metrics & Thresholds

Two threshold modes per category (both computed on the **validation** split):

| Mode | Key | Description |
|------|-----|-------------|
| Best F1 | `thr_img_f1` / `thr_px_f1` | Maximizes F1-score; balanced precision/recall |
| Low False Alarm | `thr_img_fpr` / `thr_px_fpr` | FPR ≤ 1% on normal validation images |

**Image-level thresholds** control the Normal/Anomaly decision.  
**Pixel-level thresholds** control which pixels are highlighted in the heatmap.

Reported metrics per category:

| Metric | Description |
|--------|-------------|
| Image AUROC | Area under ROC (image-level) |
| Image AUPRC | Area under precision-recall (image-level) |
| Pixel AUROC | Area under ROC (pixel-level, requires GT masks) |
| Pixel AUPRC | Area under precision-recall (pixel-level) |

---

## Troubleshooting

### "Artifacts not found" in the GUI
- Make sure you extracted the Kaggle ZIP to the `exports/` directory in the repo root
- Or change the "Artifacts directory" path in the sidebar to point to your `exports/` folder

### Test images not appearing in Browse tab
- Check that `data/visa/test/<category>/good/` and `bad/` directories exist
- Update the "Test data directory" path in the sidebar

### CUDA out of memory during training
- Reduce `batch_size` in `kaggle/config.yaml`
- Or reduce `coreset_ratio` (e.g., from 0.1 to 0.05)
- Or process fewer categories at once using `--categories candle capsules`

### Slow inference in GUI
- GPU is not required for the GUI (inference runs on CPU in ~1–2 seconds per image)
- If it's very slow, check that you have at least 8 GB RAM

### Missing pixel metrics
- Pixel AUROC/AUPRC require GT masks (`Data/Masks/Anomaly/`)
- If masks are not found in the dataset, pixel metrics are skipped automatically

---

## Citation / Credits

- VisA dataset: [SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation (ECCV 2022)](https://arxiv.org/abs/2211.10427)
- PatchCore: [Towards Total Recall in Industrial Anomaly Detection (CVPR 2022)](https://arxiv.org/abs/2106.08265)
- Feature extractor: [Wide Residual Networks (BMVC 2016)](https://arxiv.org/abs/1605.07146)
