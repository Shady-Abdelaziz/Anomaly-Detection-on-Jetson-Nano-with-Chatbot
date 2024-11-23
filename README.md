# Anomaly Detection on Jetson Nano with Chatbot

## Project Overview

This project focuses on building an **Anomaly Detection System** using advanced deep learning architectures. It utilizes **MobileNet** and **BiLSTM** (Bidirectional Long Short-Term Memory) networks to detect anomalies in real-time video feeds. The system was deployed on the **NVIDIA Jetson Nano** and uses an **IP camera** for continuous monitoring at **30 FPS**. Detected anomalies are automatically uploaded to **Google Drive** and stored in a structured database, while **LLM Gemini** is used to generate detailed reports. In addition, the system sends **email warnings** for immediate alerts, enhancing situational awareness.

Furthermore, a **chatbot** powered by **Retrieval-Augmented Generation (RAG)** has been implemented, allowing users to query and interact with the generated reports, facilitating easier access to historical data and insights.

![Jetson Nano](https://github.com/Shady-Abdelaziz/Anomaly-Detection-on-Jetson-Nano-with-Chatbot/raw/main/Jetson%20Nano/jetson.jpg?raw=true)

### Key Features:
- **Real-time Anomaly Detection**: Uses MobileNet and BiLSTM models for classifying anomalies in video streams.
- **Jetson Nano Deployment**: Efficient deployment on Jetson Nano with 30 FPS for real-time monitoring.
- **Google Drive Integration**: Automatically uploads detected anomalies and videos to Google Drive.
- **LLM Gemini Model**: Generates comprehensive reports based on detected anomalies.
- **Email Alerts**: Sends immediate email notifications for detected anomalies.
- **Retrieval-Augmented Generation (RAG)** Chatbot: Allows interactive querying of stored reports.

## Demo Video

Watch the demo video showcasing the anomaly detection system in action:

[![Anomaly Detection on Jetson Nano with Chatbot](https://img.youtube.com/vi/m2qBMKrWNKQ/0.jpg)](https://www.youtube.com/watch?v=m2qBMKrWNKQ)

## System Architecture

The architecture of the anomaly detection system involves the following components:
1. **Anomaly Detection Model**: 
   - **MobileNet** for feature extraction from video frames.
   - **BiLSTM** for sequence modeling and anomaly classification.
2. **Jetson Nano**:
   - Deploys the model and performs real-time video processing.
   - Operates at 30 FPS for continuous monitoring.
3. **Cloud Integration**:
   - Detected anomalies are uploaded to **Google Drive**, creating a structured repository of videos and reports.
4. **LLM Gemini**:
   - Generates detailed reports on the detected anomalies.
5. **Retrieval-Augmented Generation (RAG)** Chatbot:
   - Provides an interactive interface for querying and retrieving generated reports from the database.


## Setup & Installation

### Prerequisites
- **NVIDIA Jetson Nano** with **JetPack** installed.
- **IP Camera** for video streaming.
- **Google Drive** account for video storage.
- **Python 3.x** environment with necessary libraries.

### Install Dependencies
Clone the repository and install the required Python libraries:

```bash
git clone https://github.com/Shady-Abdelaziz/Anomaly-Detection-on-Jetson-Nano-with-Chatbot.git
cd Anomaly-Detection-on-Jetson-Nano-with-Chatbot
pip install -r requirements.txt
```

### Model Training
To train the anomaly detection model, run the following command:

```bash
real-time-violence-detection-mobilenet-bi-lstm.ipynb```

This will train the **MobileNet** and **BiLSTM** models on the **UCF Anomaly Detection dataset**.

### Deploying on Jetson Nano
1. Transfer the trained models to the **Jetson Nano**.
2. Configure the **IP camera** stream settings.
3. Run the real-time anomaly detection system:

```bash
python infer onnx.py
python infer using tensor rt.py

```

### Google Drive Integration
To enable automatic uploading of detected anomalies to Google Drive, you'll need to set up OAuth2 authentication. Follow the steps outlined in the [Google Drive API documentation](https://developers.google.com/drive/api/v3/quickstart-python) to create the necessary credentials and authenticate your application.

### Email Alerts
Configure the email settings in the `config.py` file to activate the email alert system, which sends notifications whenever anomalies are detected in the video stream.

### Chatbot Integration
For querying the generated reports, use the **Anomaly_Chatbot_with_streamlit.ipynb** notebook on Google Colab. Simply follow this link to access the notebook:

[Open Chatbot Notebook on Google Colab](https://colab.research.google.com/drive/1vfAdMNh-1-NzTaeimjylcJF1qUOI2gkd?authuser=1#scrollTo=bOoM8l6pTFrt)

Once you open the notebook, you can interact with the chatbot and query the stored reports from your Google Drive database.

## Acknowledgments
- **MobileNet** and **BiLSTM** models for anomaly detection.
- **UCF Anomaly Detection Dataset** for training the models.
- **NVIDIA Jetson Nano** for edge deployment.
- **Google Drive API** for cloud integration.
- **LLM Gemini** for report generation.
- **Retrieval-Augmented Generation (RAG)** for chatbot functionality.

## Conclusion

This project demonstrates an end-to-end solution for real-time anomaly detection using the Jetson Nano platform. It combines deep learning, cloud integration, and natural language processing to create a robust system for monitoring and responding to anomalous events in video streams.
