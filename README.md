# Anomaly Detection on Jetson Nano with Chatbot

## Project Overview

This project is my **graduation project** from the **National Telecommunication Institute (NTI)** as part of the **NTI 4-Month Training | Professional Training Program AI and Deep Learning Track**, **Nasr City Branch** (Aug 2024). It focuses on building an **Anomaly Detection System** leveraging **MobileNet** and **BiLSTM** (Bidirectional Long Short-Term Memory) networks for real-time anomaly detection in video feeds. Deployed on the **NVIDIA Jetson Nano**, the system utilizes an **IP camera** for continuous monitoring at **30 FPS**. Detected anomalies are uploaded to **Google Drive**, where they are stored in a structured database. **LLM Gemini** is employed to generate detailed reports, while **email warnings** are sent for immediate alerts.

A **Retrieval-Augmented Generation (RAG)** powered **chatbot** allows users to interact with and query the stored reports, simplifying access to historical anomaly data.

![Jetson Nano](https://github.com/Shady-Abdelaziz/Anomaly-Detection-on-Jetson-Nano-with-Chatbot/raw/main/Jetson%20Nano/jetson.jpg?raw=true)

### Key Features:
- **Real-time Anomaly Detection**: MobileNet and BiLSTM models classify anomalies in video streams.
- **Jetson Nano Deployment**: Efficient deployment on Jetson Nano, processing video at 30 FPS.
- **Google Drive Integration**: Detected anomalies and videos are uploaded to Google Drive.
- **LLM Gemini**: Generates detailed reports on detected anomalies.
- **Email Alerts**: Sends immediate notifications when anomalies are detected.
- **RAG Chatbot**: Allows querying of stored reports from the database.

## Demo Video

Watch the demo video showcasing the anomaly detection system:

[![Anomaly Detection on Jetson Nano with Chatbot](https://img.youtube.com/vi/m2qBMKrWNKQ/0.jpg)]([https://www.youtube.com/watch?v=m2qBMKrWNKQ](https://drive.google.com/file/d/1Hm5gzUa8XPqj8uXuWNbgL4Z5G1IwScvP/view?usp=sharing))

## System Architecture

The system consists of the following components:
1. **Anomaly Detection Model**:
   - **MobileNet**: Feature extraction from video frames.
   - **BiLSTM**: Sequence modeling and anomaly classification.
2. **Jetson Nano**: Deploys the model for real-time video processing at 30 FPS.
3. **Cloud Integration**: Anomalies are uploaded to **Google Drive**, creating a structured repository of videos and reports.
4. **LLM Gemini**: Generates comprehensive reports on detected anomalies.
5. **RAG Chatbot**: Provides an interface to query and retrieve reports stored in the cloud database.

## Setup & Installation

### Prerequisites
- **NVIDIA Jetson Nano** with **JetPack** installed.
- **IP Camera** for video streaming.
- **Google Drive** account for storing videos.
- **Python 3.x** environment with necessary libraries.

### Install Dependencies
Clone the repository and install required Python libraries:

```bash
git clone https://github.com/Shady-Abdelaziz/Anomaly-Detection-on-Jetson-Nano-with-Chatbot.git
cd Anomaly-Detection-on-Jetson-Nano-with-Chatbot
pip install -r requirements.txt
```

### Model Training
To train the anomaly detection model, run:

```bash
real-time-violence-detection-mobilenet-bi-lstm.ipynb
```

This trains the **MobileNet** and **BiLSTM** models on the **UCF Anomaly Detection dataset**.

### Deploying on Jetson Nano
1. Transfer the trained models to the **Jetson Nano**.
2. Configure the **IP camera** settings.
3. Run the real-time anomaly detection system:

```bash
python infer_onnx.py
python infer_using_tensor_rt.py
```

### Google Drive Integration
To automatically upload detected anomalies to Google Drive, set up **OAuth2 authentication** by following the steps in the [Google Drive API documentation](https://developers.google.com/drive/api/v3/quickstart-python) to create and authenticate your credentials.

### Email Alerts
Configure email settings in the `config.py` file to enable the email alert system, which will notify you whenever an anomaly is detected.

### Chatbot Integration
To query the generated reports, use the **Anomaly_Chatbot_with_streamlit.ipynb** notebook on Google Colab. Open the notebook via this link:

[Open Chatbot Notebook on Google Colab](https://colab.research.google.com/drive/1vfAdMNh-1-NzTaeimjylcJF1qUOI2gkd?authuser=1#scrollTo=bOoM8l6pTFrt)

Once opened, interact with the chatbot to query the stored reports from your Google Drive database.

## Acknowledgments
- **MobileNet** and **BiLSTM** models for anomaly detection.
- **UCF Anomaly Detection Dataset** for model training.
- **NVIDIA Jetson Nano** for edge deployment.
- **Google Drive API** for cloud storage integration.
- **LLM Gemini** for generating reports.
- **Retrieval-Augmented Generation (RAG)** for the chatbot functionality.

## Conclusion

This project provides an end-to-end solution for real-time anomaly detection on the **Jetson Nano** platform. Combining deep learning, cloud integration, and natural language processing, it offers a robust system for monitoring and responding to anomalous events in video streams.
