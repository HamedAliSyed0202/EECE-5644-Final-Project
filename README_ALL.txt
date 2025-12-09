# Breast Cancer Classification using Machine Learning and Deep Learning

This project was completed for **EECE 5644: Introduction to Machine Learning and Pattern Recognition**. It focuses on classifying breast cancer images using Support Vector Machines (SVM) and Convolutional Neural Networks (CNN), including pretrained architectures like ResNet-18 and EfficientNet-B0.

## 🔍 Project Overview

Two types of datasets were used:
- **Mammography (CBIS-DDSM)**: Includes benign and malignant cases.
- **Ultrasound (BUSI Dataset)**: Includes benign, malignant, and normal categories.

Each dataset was processed and used to train and evaluate:
- Support Vector Machines (SVM)
- Custom CNN
- ResNet-18 (Transfer Learning)
- EfficientNet-B0 (Transfer Learning)

---

## 📁 Directory Structure
Archive- contains svm for ultrasound
CNN- contains cnn for ultrasound
Mammography- contains both svm and cnn files for mammography

---

## 🛠 How to Run

1. Open MATLAB and set working directory to the folder containing this project.
2. Make sure required toolboxes are installed:
   - Image Processing Toolbox
   - Deep Learning Toolbox
   - Pretrained Model support packages (ResNet-18, EfficientNet-B0)
3. Run the scripts:
   - For ultrasound: `ultrasound_svm.m`, `ultrasound_cnn.m`
   - For mammography: `build_mammo_dataset.m`, then `mammo_svm.m`, `mammo_custom_cnn.m`, and `mammo_transfer_learning.m`

---

## ✅ Results Summary

| Model              | Dataset      | Accuracy  |
|-------------------|--------------|-----------|
| SVM               | Ultrasound   | 82.1%     |
| SVM               | Mammography  | 75.2%     |
| Custom CNN        | Ultrasound   | 85.0%     |
| Custom CNN        | Mammography  | 70.8%     |
| ResNet-18         | Mammography  | 57.7%     |
| EfficientNet-B0   | Mammography  | 64.3%     |

---

## 🙋 Author

**Syed Hammad**  
Email: Syed.ham@nrteastern.edu
Fall 2025 – Northeastern University

