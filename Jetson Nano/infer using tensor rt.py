import cv2
import numpy as np
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TensorRT model details
TRT_MODEL_PATH = r"/home/shady/Desktop/New/model_fp16.trt"
SEQUENCE_LENGTH = 10
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
THRESHOLD = 0.5
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

# TensorRT Helper Functions
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def preprocess_sequence(frames, sequence_length, image_height, image_width):
    processed_frames = []
    for frame in frames:
        frame_resized = cv2.resize(frame, (image_width, image_height))
        frame_normalized = frame_resized / 255.0
        processed_frames.append(frame_normalized)
    while len(processed_frames) < sequence_length:
        processed_frames.append(processed_frames[-1])
    return np.expand_dims(np.array(processed_frames), axis=0).astype(np.float32)

def run_inference(context, h_input, d_input, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output

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
    except Exception as e:
        print(f"Error saving video: {e}")

# Detect Anomalies
def detect_anomalies(threshold, context, h_input, d_input, h_output, d_output, stream, drive_service, drive_folder_id):
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

        if len(frame_buffer) >= SEQUENCE_LENGTH:
            input_data = preprocess_sequence(
                list(frame_buffer)[-SEQUENCE_LENGTH:], SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH
            )
            h_input[:] = input_data.ravel()
            outputs = run_inference(context, h_input, d_input, h_output, d_output, stream)
            prediction_prob = outputs[0]
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

# Initialize TensorRT Engine
engine = load_engine(TRT_MODEL_PATH)
context = engine.create_execution_context()
h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

credentials_file = r"/home/shady/Desktop/New/anomaly.json"
drive_folder_id = "1EVONoZ0UJmRWTnpQmxDwLMdypZmNOurC"
drive_service = authenticate_drive(credentials_file)

# Start anomaly detection
detect_anomalies(THRESHOLD, context, h_input, d_input, h_output, d_output, stream, drive_service, drive_folder_id)

