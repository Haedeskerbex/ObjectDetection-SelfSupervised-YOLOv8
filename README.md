# 🚀 YOLOv8 Object Detection with Self-Supervised Learning

Welcome to the **YOLOv8 Object Detection** project! This project uses YOLOv8 for object detection with a self-supervised learning loop to improve detection over time. Given hardware limitations, the training is run on the cloud, with seamless integration to local detection code via Google Cloud API. This setup allows efficient re-training and model updating with minimal manual intervention.

---

## 📝 Project Overview

This project focuses on detecting a single object class using the YOLOv8 model. We implemented a self-supervised training loop to continuously improve detection by dynamically updating the training data with confident predictions. The model is retrained in the cloud, with the latest model accessible in real-time to the local detection code via Google Drive.

## ✨ Key Features

- **Real-Time Object Detection**: Uses YOLOv8 to detect water bottles in real-time, utilizing your local webcam.
- **Self-Supervised Learning Loop**: Continuously improves model performance by dynamically retraining on new detections.
- **Cloud-Based Training**: Model training is performed in Google Colab to address hardware constraints, allowing scalable training and faster processing.
- **Google Cloud API Integration**: Synchronizes model updates between Google Drive and local detection code, ensuring the latest model is always in use.
  
---

## 🏗️ Project Structure

- **Local Files**:
  - `YOLOv8_Detection` : for object detection using real time streams from the webcam.
  - `client_secrets.json`: Google Cloud authorization file for API access.
  - `captured_frames/`: Directory where new frames are saved for self-supervised learning.
  - `.gitignore`: Ensures virtual environment and sensitive files are not tracked in Git.
  - `latest_best.pt`: Contains the latest YOLOv8 model weights.

- **Cloud Files**:
  - **Training Notebook**: A Jupyter notebook named `YOLOv8_SelfSupervised_ObjectDetection.ipynb` for cloud-based training, retraining, and data processing.

---
