# VERSION 1
# used with nabils_code.py version-3

import os
import tempfile
import whisper

WHISPER_MODEL_NAME = "base"

def load_whisper_model():
    return whisper.load_model(WHISPER_MODEL_NAME)

def transcribe_audio_file(audio_file, whisper_model):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path)
        text = result["text"].strip()

        os.remove(tmp_path)
        return text

    except Exception as e:
        return f"__ERROR__: {str(e)}"
