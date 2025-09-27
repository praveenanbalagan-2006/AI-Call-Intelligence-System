import os
import uuid
import time
import json
from flask import Flask, request, jsonify, render_template, render_template_string
from pydub import AudioSegment
from faster_whisper import WhisperModel
# Import Firestore functions from db.py
from db import initialize_firestore, save_analysis_to_firestore, fetch_all_analysis_records

# Initialize Flask App
app = Flask(__name__)

# Define the upload folder and ensure it exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check for ffmpeg and warn the user
try:
    from pydub.utils import get_prober_name
    get_prober_name()
except Exception as e:
    print("WARNING: FFmpeg or Avconv is not found. Audio conversion may fail. Please ensure it is installed and in your PATH.")
    print(f"Error details: {e}")

# Load the Whisper model for speech-to-text
# The 'large-v2' model is the most accurate for multiple languages.
# It is capable of both transcription and direct English translation.
model_size = "large-v2"
try:
    # Use the class constructor directly
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    model = None

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Translator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen flex items-center justify-center p-4">

    <div class="bg-gray-800 rounded-2xl shadow-2xl p-8 max-w-2xl w-full">
        <h1 class="text-3xl font-bold text-center text-indigo-400 mb-6">
            Speech-to-Text Translator
        </h1>
        <p class="text-center text-gray-400 mb-8">
            Upload an audio file in any language to get its original transcription and an English translation.
        </p>

        <!-- File Upload Form -->
        <form id="upload-form" class="flex flex-col gap-6" enctype="multipart/form-data">
            <div class="flex flex-col">
                <label for="audio-file" class="text-sm font-medium text-gray-300 mb-2">Select Audio File</label>
                <input type="file" id="audio-file" name="audio_file" accept="audio/*" class="w-full p-3 rounded-lg bg-gray-700 text-gray-100 border border-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer transition duration-200">
            </div>

            <button type="submit" id="submit-btn" class="bg-indigo-600 hover:bg-indigo-700 transition duration-300 text-white font-bold py-3 px-6 rounded-xl shadow-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50">
                Transcribe & Translate
            </button>
        </form>

        <!-- Loading Indicator and Error Message -->
        <div id="status-message" class="hidden mt-6 text-center text-gray-400">
            <div id="loading-spinner" class="hidden">
                <svg class="animate-spin h-6 w-6 mx-auto text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <p class="mt-2">Processing audio...</p>
            </div>
            <p id="error-message" class="text-red-400 font-semibold"></p>
        </div>

        <!-- Result Section -->
        <div id="result-container" class="hidden mt-8">
            <h2 class="text-xl font-semibold text-indigo-300 mb-4">Transcription:</h2>
            <div id="transcription-text" class="bg-gray-700 text-gray-200 p-4 rounded-xl shadow-inner whitespace-pre-wrap mb-6"></div>
            
            <h2 class="text-xl font-semibold text-indigo-300 mb-4">English Translation:</h2>
            <div id="translation-text" class="bg-gray-700 text-gray-200 p-4 rounded-xl shadow-inner whitespace-pre-wrap"></div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const form = document.getElementById('upload-form');
            const audioFile = document.getElementById('audio-file');
            const submitBtn = document.getElementById('submit-btn');
            const statusMessage = document.getElementById('status-message');
            const loadingSpinner = document.getElementById('loading-spinner');
            const errorMessage = document.getElementById('error-message');
            const resultContainer = document.getElementById('result-container');
            const transcriptionTextDiv = document.getElementById('transcription-text');
            const translationTextDiv = document.getElementById('translation-text');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const file = audioFile.files[0];
                
                if (!file) {
                    errorMessage.textContent = 'Please select an audio file to upload.';
                    statusMessage.classList.remove('hidden');
                    return;
                }

                submitBtn.disabled = true;
                loadingSpinner.classList.remove('hidden');
                errorMessage.textContent = '';
                resultContainer.classList.add('hidden');
                statusMessage.classList.remove('hidden');

                const formData = new FormData();
                formData.append('audio_file', file);

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    
                    if (response.ok) {
                        transcriptionTextDiv.textContent = result.transcription;
                        translationTextDiv.textContent = result.translation;
                        resultContainer.classList.remove('hidden');
                        statusMessage.classList.add('hidden');
                    } else {
                        errorMessage.textContent = `Error: ${result.error || 'An unknown error occurred.'}`;
                        statusMessage.classList.remove('hidden');
                    }
                } catch (error) {
                    errorMessage.textContent = 'An unexpected error occurred. Please try again.';
                    statusMessage.classList.remove('hidden');
                } finally {
                    submitBtn.disabled = false;
                    loadingSpinner.classList.add('hidden');
                }
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Renders the main page with the embedded HTML template."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Handles the audio file upload, transcription, and translation."""
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        start_time = time.time()
        
        # Save the uploaded file to a temporary path
        file_ext = os.path.splitext(file.filename)[1]
        temp_filename = f"{uuid.uuid4().hex}{file_ext}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_filepath)

        transcription = ""
        translation = ""
        try:
            # Convert audio to a format compatible with faster-whisper if necessary
            audio_path_to_transcribe = temp_filepath
            if file_ext.lower() not in ['.mp3', '.wav', '.ogg', '.flac']:
                audio_file = AudioSegment.from_file(temp_filepath)
                converted_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}.mp3")
                audio_file.export(converted_path, format="mp3")
                audio_path_to_transcribe = converted_path

            # Use faster-whisper to get the original transcription and detected language
            segments_orig, info_orig = model.transcribe(audio_path_to_transcribe, beam_size=5)
            detected_lang = info_orig.language
            transcribed_text = " ".join([segment.text for segment in segments_orig])

            # Now, use the same model to translate to English
            segments_trans, info_trans = model.transcribe(audio_path_to_transcribe, beam_size=5, task="translate")
            translated_text = " ".join([segment.text for segment in segments_trans])

            transcription = transcribed_text
            translation = translated_text

            print("--------------------------------------------------")
            print(f"Transcription and Translation Complete from file: {temp_filepath}")
            print(f"Detected Language: {detected_lang}")
            print(f"Time Taken: {round(time.time() - start_time, 2)} seconds")
            print("--------------------------------------------------")

        except Exception as e:
            # Clean up the temporary file on error
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the temporary files
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            if 'converted_path' in locals() and os.path.exists(converted_path):
                os.remove(converted_path)

        return jsonify({
            "transcription": transcription,
            "translation": translation,
            "detected_language": detected_lang
        })
    
    return jsonify({"error": "An unknown error occurred during file upload"}), 500

if __name__ == '__main__':
    # Initialize Firestore before running the app
    # You must provide your app_id and firebase_config_json here
    APP_ID = "my_app_id"  # Change as needed
    FIREBASE_CONFIG_JSON = json.dumps({"projectId": "your_project_id"})  # Change as needed
    initialize_firestore(APP_ID, FIREBASE_CONFIG_JSON)
    app.run(debug=True)
# Dashboard route to display records from Firestore
@app.route('/dashboard')
def dashboard():
    records = fetch_all_analysis_records(APP_ID)
    return render_template('dashboard.html', records=records)
