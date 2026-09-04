# NeuroScan — Brain Tumor Detection System

A computer vision tool that classifies brain MRI scans into four categories — glioma, meningioma, pituitary tumor, or no tumor — using a CNN trained on real-world MRI data, served through an interactive Streamlit app.

**[Live demo](#)** · **[Report a bug](#)**

---

## Overview

NeuroScan takes a single MRI slice as input and returns a classification with a confidence score across all four classes. It's built as an educational computer-vision demo for medical image analysis, not a diagnostic tool — see the [disclaimer](#disclaimer) below.

## Features

- Uploads a JPG/PNG MRI scan and classifies it in real time
- CNN model quantized to TFLite for fast, lightweight inference
- Confidence score plus a full probability breakdown across all four classes
- Clean, responsive Streamlit interface

## Tech stack

| Layer | Tools |
|---|---|
| Model | TensorFlow / Keras (CNN), quantized to TFLite |
| Image processing | OpenCV, Pillow, NumPy |
| App / UI | Streamlit |
| Deployment | Streamlit Cloud |

## Model details

- Trained on 6,000+ MRI scans using normalization and augmentation
- Hyperparameters tuned via grid search for accuracy and generalization
- **85.8% accuracy** on the held-out real-world test set
- Evaluated with standard classification metrics and validation techniques
- Output classes: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`

## Screenshots

<!-- Add screenshots or a short GIF of the app here, e.g.: -->
<!-- ![NeuroScan UI](assets/screenshot.png) -->

## Getting started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/neuroscan.git
cd neuroscan
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload an MRI scan (JPG or PNG) and click **Analyze scan**.

### Requirements

Make sure `requirements.txt` includes at least:

```
streamlit
tensorflow
numpy
opencv-python-headless
Pillow
```

## Project structure

```
neuroscan/
├── app.py                              # Streamlit app
├── brain_tumor_model_quantized.tflite  # Trained, quantized CNN model
├── requirements.txt
└── README.md
```

## Disclaimer

NeuroScan is an educational, AI-assisted demonstration of computer-vision techniques applied to medical imaging. It is **not a certified diagnostic tool** and should never be used as a substitute for evaluation by a qualified medical professional. Always consult a doctor for diagnosis and treatment.

## Author

**Arpan Kumar Mallik**
[LinkedIn](#) · [GitHub](#) · [Portfolio](#) · arpanmallik.careers@gmail.com

## License

Distributed under the MIT License. See `LICENSE` for details.
