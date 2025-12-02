#Chest X-ray Pneumonia Detection using DenseNet121 (Deep Learning Project)

This project implements an end-to-end deep learning pipeline for detecting Pneumonia from chest X-ray images using Transfer Learning (DenseNet121), data augmentation, class weighting, fine-tuning, and explainability using GradCAM.
A full GUI is deployed using Gradio.

🚀 Features

DenseNet121 pretrained on ImageNet

Fine-tuning last layers

Strong augmentation

Class-weight balancing for imbalanced dataset

GradCAM heatmap visualization

Gradio GUI for real-time predictions

Full evaluation: Accuracy, F1, ROC-AUC

End-to-end reproducible project

📁 Dataset

Chest X-ray Pneumonia
Source: Kaggle (MOONEY Dataset)

Contains:

Normal: 1341 images

Pneumonia: 3875 images
Split into:

Train

Validation
🧠 Model Architecture

DenseNet121 base model

Global Average Pooling

Dense layer (256 units, L2 regularization)

Dropout 0.4

Sigmoid output

🎨 Explainability (GradCAM)

The project includes a fine-tuned GradCAM implementation showing heatmaps highlighting infected lung areas.
