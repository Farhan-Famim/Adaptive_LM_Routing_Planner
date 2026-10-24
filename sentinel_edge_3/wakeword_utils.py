import numpy as np
import openwakeword
from openwakeword.model import Model

WAKEWORD_MODEL_KEY = "hey jarvis"

def load_wakeword_model():
    openwakeword.utils.download_models()
    return Model(inference_framework="onnx", vad_threshold=0.5)

def bytes_to_int16_pcm(audio_bytes):
    return np.frombuffer(audio_bytes, dtype=np.int16)

def detect_wakeword_from_chunk(audio_bytes, wake_model, threshold=0.3):
    try:
        pcm = bytes_to_int16_pcm(audio_bytes)
        prediction = wake_model.predict(pcm)
        score = prediction.get(WAKEWORD_MODEL_KEY, 0.0)
        return score >= threshold, score
    except Exception:
        return False, 0.0