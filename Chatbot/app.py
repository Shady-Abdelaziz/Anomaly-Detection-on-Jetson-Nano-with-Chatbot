import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import glob

# Initialize API and Environment
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyBaHgCrTHdVfDWNzgUhBUYuGyKgElLH6cQ')  # Use env variable if available
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_ROLE = 'Anomaly Detection'
AI_AVATAR_ICON = '✨'
REPORT_DIR = "/content/drive/MyDrive/Anomaly_Chatbot/drive_downloads"

# Create data/ folder if it doesn't already exist
os.makedirs('data/', exist_ok=True)

# Load all reports
def load_reports():
    file_paths = glob.glob(f"{REPORT_DIR}/*.txt")
    reports = {os.path.basename(fp): open(fp, 'r').read() for fp in file_paths}
    return reports

reports = load_reports()

st.write('# Anomaly Detection with Chat ')

# Configure safety settings for Gemini model
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Initialize the generative model
st.session_state.model = genai.GenerativeModel(
    model_name='gemini-1.5-pro',
    safety_settings=safety_settings
)

st.session_state.chat = st.session_state.model.start_chat()

# Inject report data into the chat context
def augment_context_with_reports(prompt, reports):
    # Combine relevant sections of reports with the user input
    context = "\n\n".join(
        [f"Report: {name}\nContent: {content}" for name, content in reports.items()]
    )
    return f"{context}\n\nUser Query: {prompt}"

# Retry logic without delay
def send_message_with_retry(chat, prompt, retries=3):
    """Send a message to the chat model with retry logic for handling quota exhaustion."""
    for attempt in range(retries):
        try:
            return chat.send_message(prompt, stream=True)
        except Exception as e:
            if "429" in str(e):  # Quota exhaustion error
                st.warning("Quota exceeded. Retrying...")
            else:
                st.error(f"Error: {e}")
                raise
    st.error("Quota exceeded after multiple retries.")
    raise Exception("Quota exceeded after multiple retries.")

# Handle user input
if prompt := st.chat_input('Your message here...'):
    with st.chat_message('user'):
        st.markdown(prompt)

    # Prepare augmented input
    augmented_prompt = augment_context_with_reports(prompt, reports)

    # Send the message to AI and display the response
    try:
        response = send_message_with_retry(st.session_state.chat, augmented_prompt)

        # Display AI response in chat
        with st.chat_message(name=MODEL_ROLE, avatar=AI_AVATAR_ICON):
            message_placeholder = st.empty()
            full_response = ''
            for chunk in response:
                for ch in chunk.text.split(' '):
                    full_response += ch + ' '
                    message_placeholder.write(full_response + '▌')
            message_placeholder.write(full_response)

    except Exception as e:
        if "Quota exceeded" in str(e):
            st.error("The system is temporarily unavailable due to API quota limits. Please try again later.")
        else:
            st.error(f"Failed to fetch response: {e}")
