# 🍇 Grape Disease Detection with Attention Mechanism & GradCAM

> **Artificial Intelligence Project | East West University, Dhaka, Bangladesh**  
> **Author:** Faria Azad Anita

---

## 📌 Overview

Grape diseases significantly reduce crop quality and yield. This project develops a **deep learning system** for classifying grape leaf diseases into **4 classes** using multiple CNN architectures, a custom **Channel Attention Mechanism**, and **GradCAM visualization** to interpret model predictions.

Multiple state-of-the-art models are compared to find the best performer, with attention layers and explainability tools added for deeper insight.

---

## 🎯 Disease Classes

| Label | Class |
|-------|-------|
| 0 | Black Rot |
| 1 | Esca (Black Measles) |
| 2 | Leaf Blight |
| 3 | Healthy |

---

## 🏗️ Models Compared

| Model | Type | Description |
|-------|------|-------------|
| MobileNetV3Large | Pretrained CNN | Lightweight, efficient transfer learning |
| ResNet50 | Pretrained CNN | Deep residual network |
| VGG16 | Pretrained CNN | Classic deep CNN |
| CustomCNN | From scratch | Custom architecture with attention |
| CustomCNN + Pooling | From scratch | Custom CNN with custom pooling layers |

---

## 🔬 Key Features

### 1. Custom Channel Attention Mechanism
```
ChannelAttention(in_planes, ratio=16)
├── AdaptiveAvgPool2d
├── AdaptiveMaxPool2d
├── FC layers (squeeze & excitation)
└── Sigmoid activation
```
Forces the model to focus on the **most important feature channels**, improving disease boundary detection.

### 2. GradCAM Visualization
- Generates **heatmaps** showing which regions of the leaf the model focuses on
- Helps explain **why** the model made a prediction
- Critical for building trust in AI-based agricultural tools

---

## 🔬 Methodology

1. **Dataset Loading** — Grape Disease Dataset from Kaggle (4 classes)
2. **Preprocessing** — Resize to 224×224, normalize, augmentation
3. **Model Training** — Train and compare 5 different architectures
4. **Attention Integration** — Add Channel Attention to custom CNN
5. **GradCAM** — Visualize and interpret model decisions
6. **Evaluation** — Confusion matrix, classification report, accuracy

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Deep Learning | Python, PyTorch, torchvision |
| Explainability | GradCAM |
| Data Processing | NumPy, OpenCV, Matplotlib, Pandas |
| Environment | Google Colab, Kaggle |
| Models | MobileNetV3, ResNet50, VGG16, CustomCNN |

---

## 📂 Repository Structure

```
Grape-Disease-Attention-GradCAM/
│
├── 366(MobileNetV3Large).ipynb
├── 366(resnet50).ipynb
├── 366_VGG16.ipynb
├── 366CustomCNN.ipynb
├── CustomCNNwithPooling.ipynb
├── Attention_layer+Gradcam.ipynb    ← Main notebook with attention + GradCAM
└── README.md
```

---

## 📥 Dataset

**Grape Disease Dataset (Original)**  
📦 [Kaggle — rm1000/grape-disease-dataset-original](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original)

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Input Size | 224×224 |
| num_classes | 4 |
| Loss Function | CrossEntropyLoss |
| Augmentation | RandomCrop, Flip, Rotation, Perspective |
| Device | GPU (CUDA) / CPU |

---

## 📊 Data Augmentation Pipeline

```python
transforms.RandomResizedCrop(224)
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.2)
transforms.RandomRotation(15)
transforms.RandomAffine(translate=(0.1, 0.1))
transforms.RandomPerspective(distortion_scale=0.2)
```

---

## 👩‍💻 Author

**Faria Azad Anita**  
Undergraduate Student, CSE — East West University, Dhaka, Bangladesh  
📧 faria.azad.anita2001@gmail.com  
🔗 [GitHub](https://github.com/Anita-faria)

---

<p align="center">Made with ❤️ for agricultural AI research</p>
