# Version 2.0
# audio output is generated from the device speaker directly, not the browser

import os
import requests
from tqdm import tqdm
import sounddevice as sd
import soundfile as sf
import io
import uuid

# Define exact model files
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

MODELS_DIR = "tts_models"
MODEL_PATH = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(MODELS_DIR, "voices-v1.0.bin")

_kokoro_instance = None

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024 # 1 Megabyte
    
    with open(dest_path, "wb") as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def get_kokoro_model():
    global _kokoro_instance
    if _kokoro_instance is not None:
        return _kokoro_instance
        
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        print("Please install kokoro-onnx: pip install kokoro-onnx soundfile")
        return None

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading Kokoro TTS model (this is a one-time ~300MB download)...")
        download_file(MODEL_URL, MODEL_PATH)
        
    if not os.path.exists(VOICES_PATH):
        print(f"Downloading Kokoro TTS voices file...")
        download_file(VOICES_URL, VOICES_PATH)
        
    print("Loading Kokoro ONNX model...")
    _kokoro_instance = Kokoro(MODEL_PATH, VOICES_PATH)
    print("Kokoro model loaded successfully!")
    return _kokoro_instance

def generate_speech_bytes(text, voice="af_heart", speed=1.0):
    """
    Generates speech and returns the WAV file bytes for playback in Streamlit.
    """
    kokoro = get_kokoro_model()
    if not kokoro:
        return None
        
    print(f"Generating TTS for text length {len(text)}...")
    try:
        samples, sample_rate = kokoro.create(
            text, voice=voice, speed=speed, lang="en-us"
        )
        
        # Save to an in-memory buffer as WAV
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        print(f"Error generating TTS: {e}")
        return None


def play_speech_bytes(audio_bytes):
    """
    Play WAV bytes locally through the computer speaker and block until playback finishes.
    """
    try:
        buffer = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(buffer, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
        return True
    except Exception as e:
        print(f"Error playing TTS audio locally: {e}")
        return False
