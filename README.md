# Anomaly Detection on Jetson Nano with Chatbot

![Project Banner]([https://via.placeholder.com/800x200.png?text=Anomaly+Detection+on+Jetson+Nano](https://github.com/Shady-Abdelaziz/Anomaly-Detection-on-Jetson-Nano-with-Chatbot/blob/main/Jetson%20Nano/jetson.jpg)) <!-- Replace with an actual banner image -->

## Overview

This project is a real-time **Anomaly Detection System** designed to monitor video feeds using **NVIDIA Jetson Nano** and an **IP camera**. It leverages advanced AI architectures and state-of-the-art models to identify anomalies, generate reports, and enable interactive querying through a chatbot.

### Key Features

- **Anomaly Detection**:
  - Leveraged **MobileNet** and **BiLSTM** architectures.
  - Trained on the **UCF Anomaly Detection dataset**.
  - Achieved real-time monitoring at **30 FPS**.
- **Deployment**:
  - Utilized **NVIDIA Jetson Nano** for edge inference.
  - Integrated with an **IP camera** for continuous video streaming.
- **Data Management**:
  - Automatically uploads detected anomalies to **Google Drive**.
  - Creates a structured database of anomaly videos.
- **Report Generation**:
  - Uses **Gemini LLM** to generate detailed reports of detected events.
  - Enables easy retrieval and analysis of reports.
- **Alerts**:
  - Sends **email warnings** for immediate notification of anomalies.
- **Interactive Chatbot**:
  - Implemented a chatbot using **Retrieval-Augmented Generation (RAG)**.
  - Allows users to interactively query stored reports for detailed insights.

---

## Demo Video

Watch the demo video explaining the entire project in action:


## Demo Video

Watch the demo video explaining the entire project:

<video width="800" controls>
  <source src="https://github.com/Shady-Abdelaziz/Anomaly-Detection-on-Jetson-Nano-with-Chatbot/blob/main/Anomaly%20Detection%20Demo.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>


---

## System Architecture

### Components

1. **Model**:
   - Backbone: **MobileNet** for feature extraction.
   - Sequence Learning: **BiLSTM** for temporal anomaly detection.

2. **Deployment**:
   - **Jetson Nano** for edge AI.
   - **IP Camera** for real-time video feeds.

3. **Reporting**:
   - **LLM Gemini Model** for video summarization and report generation.

4. **Alerts**:
   - **Email API** for warning notifications.

5. **Chatbot**:
   - **RAG Framework** for querying generated reports.

### Workflow

1. Real-time video streaming is processed using the **MobileNet-BiLSTM** pipeline on Jetson Nano.
2. Detected anomalies are:
   - Saved as short video clips.
   - Uploaded to **Google Drive**.
3. Anomaly clips are analyzed, and reports are generated using **Gemini LLM**.
4. **Email alerts** notify users of detected anomalies.
5. Reports are stored in a database, accessible via a **chatbot**.

---

## Installation and Setup

### Prerequisites

- NVIDIA Jetson Nano (configured and updated with JetPack)
- Python 3.8+
- IP Camera with RTSP support
- Google Drive API credentials
- SMTP server credentials for email alerts

### Dependencies

Install the required Python libraries:

```bash
pip install tensorflow opencv-python pillow requests langchain transformers

