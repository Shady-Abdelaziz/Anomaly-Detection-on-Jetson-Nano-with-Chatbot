# Anomaly Detection on Jetson Nano with Chatbot

This project implements an anomaly detection system capable of identifying multiple classes of anomalies (e.g., fighting, robbery, vandalism) using a pre-trained model deployed on a Jetson Nano. The system integrates with a chatbot for reporting incidents and generating summaries via NLP.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Hardware Requirements](#hardware-requirements)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview
This repository combines computer vision, anomaly detection, and natural language processing to detect and report unusual activities. Leveraging a Jetson Nano, the system processes live video streams, identifies anomalies, and triggers chatbot-generated reports using an LLM (Large Language Model).

### Key Highlights
- Real-time video anomaly detection.
- Supports 60 FPS for inference.
- Generates 30-second video clips of detected anomalies.
- Uses an LLM to summarize video content and produce detailed incident reports.
- Compatible with Zavio cameras.

## Features
- **Multi-Class Anomaly Detection**: Identifies activities like abuse, assault, robbery, vandalism, and more.
- **Real-time Performance**: Achieve 60 FPS for live video inference.
- **30-second Video Clips**: Saves and uploads a 30-second clip of detected events.
- **Chatbot Integration**: Sends alerts and generates reports for detected anomalies.

## Setup and Installation

### Prerequisites
- Jetson Nano with a compatible OS installed.
- Zavio camera (or any compatible camera source).
- Python 3.8 or higher.
- Required Python libraries: OpenCV, TensorFlow, PyTorch, NLTK, and others listed in `requirements.txt`.

### Installation Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/Shady-Abdelaziz/Anomaly-Detection-on-Jetson-Nano-with-Chatbot.git
   cd Anomaly-Detection-on-Jetson-Nano-with-Chatbot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Jetson Nano for real-time video input:
   - Ensure your camera is connected and accessible.

4. Run the detection script:
   ```bash
   python detect_anomalies.py
   ```

## Usage

After installation, you can start the anomaly detection system. When anomalies are detected, a report will be generated, and a chatbot will notify you with a summary of the event.

### Run the chatbot integration:
```bash
python chatbot_integration.py
```

### Configuration

Adjust the configuration settings in `config.json` to set video input, threshold for anomaly detection, and other parameters.

## Model Details

This project uses a pre-trained model for anomaly detection. The model is capable of recognizing various categories such as:

- Abuse
- Arrest
- Arson
- Assault
- Burglary
- Explosion
- Fighting
- Normal
- RoadAccidents
- Robbery
- Shooting
- Shoplifting
- Stealing
- Vandalism

### Loading the model:
```python
from keras.models import load_model
model = load_model('model.keras')
```

## Hardware Requirements
- **Jetson Nano** or any compatible edge device.
- **Camera**: Zavio camera or another compatible camera for video capture.

## Future Enhancements
- Integrate more anomaly categories.
- Improve model accuracy with more diverse datasets.
- Extend chatbot capabilities to handle more complex queries.

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
