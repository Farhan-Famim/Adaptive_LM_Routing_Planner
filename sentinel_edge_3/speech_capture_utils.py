import numpy as np
import io
import wave
import av
from streamlit_webrtc import AudioProcessorBase


def compute_audio_level(audio_chunk: np.ndarray) -> float:
    if len(audio_chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32)))))


def pcm_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_array.astype(np.int16).tobytes())

    buffer.seek(0)
    return buffer.read()


class InMemoryAudioFile:
    def __init__(self, audio_bytes: bytes):
        self._audio_bytes = audio_bytes

    def read(self):
        return self._audio_bytes


class SpeechCaptureProcessor(AudioProcessorBase):
    def __init__(self):
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        self.completed_audio = None

        self.speech_threshold = 25
        self.silence_limit = 4

        self.last_audio_level = 0.0
        self.frames_seen = 0

    def recv_queued(self, frames):
        for frame in frames:
            frame_array = frame.to_ndarray()

            # Convert to mono safely
            if frame_array.ndim == 2:
                if frame_array.shape[0] <= 2:
                    frame_array = frame_array.mean(axis=0)
                else:
                    frame_array = frame_array.mean(axis=1)
            else:
                frame_array = frame_array.reshape(-1)

            pcm_chunk = np.asarray(frame_array, dtype=np.int16)
            audio_level = compute_audio_level(pcm_chunk)

            self.last_audio_level = audio_level
            self.frames_seen += 1
            print(f"[SpeechCapture] frames_seen={self.frames_seen}, audio_level={audio_level:.2f}")

            if audio_level > self.speech_threshold:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_buffer = []
                    print(f"[SpeechCapture] speech started at level={audio_level:.2f}")

                self.speech_buffer.append(pcm_chunk)
                self.silence_counter = 0

            elif self.is_speaking:
                self.speech_buffer.append(pcm_chunk)
                self.silence_counter += 1

                if self.silence_counter >= self.silence_limit:
                    if len(self.speech_buffer) > 0:
                        self.completed_audio = np.concatenate(self.speech_buffer)
                        print(f"[SpeechCapture] speech ended, buffered_chunks={len(self.speech_buffer)}")

                    self.speech_buffer = []
                    self.is_speaking = False
                    self.silence_counter = 0

        return frames