# 🐍 Python Kivy ML App

A modular Python-Kivy application combining UI features, cryptographic utilities, and deep learning tools for image and video processing.

---

## ✨ Features

- 🔐 **Security**
  - Custom password hashing & verification
  - Image and text encryption using PyCryptodome

- 🖼️ **UI & UX**
  - Cross-platform support (Windows/Linux)
  - Basic image operations (preview, selection, annotation)
  - Responsive UI using KivyMD
  - Lazy image loading with pagination

- 🤖 **Machine Learning**
  - Image classification with support for:
    - MobileNetV2 / V3
    - ResNet / ResNeXt
    - EfficientNet / V2
    - VGG / AlexNet
  - YOLOv8-11-based object detection with Ultralytics
  - Custom training pipeline with TensorBoard logging
  - Manual model selection, fine-tuning, and evaluation

- 🎥 **Video**
  - Live camera preview (OpenCV)
  - Real-time detection on frames

- 📁 **Project Management**
  - Multi-project support (datasets, models, configs)
  - Image copying between projects
  - Project-based dataset split and model organization

---

## 🧪 Getting Started

### ✅ Requirements

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Includes:

- `kivy`, `kivymd`
- `torch`, `torchvision`
- `ultralytics`, `opencv-python`
- `numpy`, `scikit-learn`, `pycryptodomex`
- `labelImg`, `pre_commit`

---

### 🧠 Torch & Ultralytics Notes

- Ensure Torch and torchvision are installed with correct CUDA or CPU builds for your environment.

- Ultralytics YOLO configuration:

  View current settings:

  ```bash
  yolo settings
  ```
  Modify runs directory path:

  ```bash
  yolo settings runs_dir=/path/to/runs
  ```
  
### 📌 TODO

- Improve UI/UX for multi-project workflows
- Support `.yaml` config export/import
- Add face detection and recognition modules
- Provide Docker image for easier deployment
- Extend install scripts for both Torch and TensorFlow
- Full support for model hyperparameter tuning

---

### 🧠 Tips

TensorBoard logs are saved per project. Launch from app or CLI:

```bash
tensorboard --logdir=./runs
```
`.exe` builds supported via buildozer or manual packaging (Windows/Linux)

---

### 📂 File Structure (Example)

```bash
projects/
└── ProjectName/
    ├── models/
    ├── train/
    ├── val/
    ├── test/
    └── configs/
```

---

## 🤝 Contributions
TODO

---

## 📜 License
TODO
