import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
import google.generativeai as genai
import time

# Configuration for generative AI
genai.configure(api_key="AIzaSyBzZdRb7yO0YpdiBCQKU5gvoVXby4CUftA")

# ONNX model details
ONNX_PATH = r"/home/shady/Desktop/New/90.onnx"
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
THRESHOLD = 0.7
ANOMALY_TRIGGER_SECONDS = 3
RECORD_DURATION_SECONDS = 15
SAVE_BASE_DIR = r"/home/shady/Desktop/New/Anomaly_clips"

# Email details
EMAIL_ADDRESS = "anomalydetection81@gmail.com"
EMAIL_PASSWORD = "qitv fkix vkev bdkl"
WARNING_RECIPIENT = "shady.sakr52@gmail.com"

os.makedirs(SAVE_BASE_DIR, exist_ok=True)

# Google Drive Authentication
def authenticate_drive(credentials_path):
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build('drive', 'v3', credentials=creds)

# Google Drive Upload
def upload_to_drive(file_path, service, folder_id=None):
    try:
        file_metadata = {'name': os.path.basename(file_path)}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        media = MediaFileUpload(file_path, resumable=True)
        uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Uploaded to Google Drive: {file_path}, File ID: {uploaded_file.get('id')}")
    except Exception as e:
        print(f"Error uploading file {file_path}: {e}")

# Send Email Notification
def send_report_email(report_path):
    try:
        subject = "Anomaly Report Uploaded"
        body = f"A detailed report of the detected anomaly has been generated. Report path: {report_path}"

        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = WARNING_RECIPIENT
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, WARNING_RECIPIENT, msg.as_string())

        print(f"Report email sent to {WARNING_RECIPIENT}.")
    except Exception as e:
        print(f"Failed to send report email: {e}")

# Generate Report using LLM
def generate_report(video_file):
    try:
        video_file = genai.upload_file(path=video_file)
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed.")

        prompt = """Provide a detailed analysis of events in the video, highlighting any anomalies."""
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
        return response.text
    except Exception as e:
        print(f"Error generating report: {e}")
        return "Failed to generate report."

# Preprocess Frame (Updated to handle individual frames)
def preprocess_frame(frame, image_height, image_width):
    frame_resized = cv2.resize(frame, (image_width, image_height))
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(frame_normalized, axis=0).astype(np.float32)

# Save Video Clip and Process
def save_video_and_process(clip_path, frames, fps, frame_size, drive_service, drive_folder_id, executor):
    try:
        # Save video synchronously
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, fps, frame_size)
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Saved video: {clip_path}")
        
        # Call handle_clip only after video is completely saved
        handle_clip(clip_path, drive_service, drive_folder_id, executor)
    except Exception as e:
        print(f"Error saving video: {e}")

# Handle Clip
def handle_clip(clip_path, drive_service, drive_folder_id, executor):
    try:
        # Generate report and upload in a background thread
        def process_clip():
            report_content = generate_report(clip_path)
            report_path = clip_path.replace(".mp4", "_report.txt")
            with open(report_path, "w") as report_file:
                report_file.write(report_content)
            upload_to_drive(clip_path, drive_service, drive_folder_id)
            upload_to_drive(report_path, drive_service, drive_folder_id)
            send_report_email(report_path)

        executor.submit(process_clip)
    except Exception as e:
        print(f"Error handling clip {clip_path}: {e}")

# Detect Anomalies
def detect_anomalies(threshold, session, drive_service, drive_folder_id):
    cap = cv2.VideoCapture("rtsp://admin:admin@192.168.251.3/video.mjpg")
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_buffer = deque(maxlen=RECORD_DURATION_SECONDS * fps)
    anomaly_start_time = None
    recording_start_time = None
    recording_in_progress = False

    executor = ThreadPoolExecutor(max_workers=3)

    print("Starting video capture...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        frame_buffer.append(frame)
        is_anomaly = False

        if len(frame_buffer) >= 1:  # No sequence length, use single frame
            input_data = preprocess_frame(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
            outputs = session.run(None, {'args_0': input_data})[0]
            prediction_prob = outputs[0][0]
            is_anomaly = prediction_prob >= threshold

            if is_anomaly:
                if anomaly_start_time is None:
                    anomaly_start_time = datetime.now()
                elif (datetime.now() - anomaly_start_time).total_seconds() >= ANOMALY_TRIGGER_SECONDS and not recording_in_progress:
                    print("Recording anomaly...")
                    recording_in_progress = True
                    recording_start_time = datetime.now()

            if recording_in_progress:
                if (datetime.now() - recording_start_time).total_seconds() >= RECORD_DURATION_SECONDS:
                    recording_in_progress = False
                    anomaly_start_time = None
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_path = os.path.join(SAVE_BASE_DIR, f"anomaly_{timestamp}.mp4")
                    frame_list = list(frame_buffer)
                    frame_size = (frame_list[0].shape[1], frame_list[0].shape[0])
                    executor.submit(save_video_and_process, clip_path, frame_list, fps, frame_size, drive_service, drive_folder_id, executor)

        label = "Anomaly" if is_anomaly else "Normal"
        font_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2, cv2.LINE_AA)
        cv2.imshow("Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()

# Initialize ONNX model session
onnx_session = ort.InferenceSession(ONNX_PATH)
credentials_file = r"/home/shady/Desktop/New/anomaly.json"
drive_folder_id = "1EVONoZ0UJmRWTnpQmxDwLMdypZmNOurC"
drive_service = authenticate_drive(credentials_file)

# Start anomaly detection
detect_anomalies(THRESHOLD, onnx_session, drive_service, drive_folder_id)

