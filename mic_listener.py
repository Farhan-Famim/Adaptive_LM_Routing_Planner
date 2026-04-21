# Version 1.0
# Takes audio input from the mic of the device. Automated voice input.

import queue
import time
import numpy as np
import sounddevice as sd


class MicUtteranceListener:
    def __init__(
        self,
        sample_rate=16000,
        channels=1,
        blocksize=1024,
        speech_threshold=80,
        silence_limit=20,
        max_record_seconds=12,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.speech_threshold = speech_threshold
        self.silence_limit = silence_limit
        self.max_record_seconds = max_record_seconds

        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MicListener] status: {status}")
        self.audio_queue.put(indata.copy())

    def _compute_audio_level(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return 0.0
        chunk = chunk.astype(np.float32)
        return float(np.sqrt(np.mean(np.square(chunk))))

    def capture_single_utterance(self):
        speech_buffer = []
        is_speaking = False
        silence_counter = 0
        start_time = time.time()

        print("[MicListener] Waiting for speech...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="int16",
            callback=self._audio_callback,
        ):
            while True:
                if time.time() - start_time > self.max_record_seconds:
                    print("[MicListener] Timeout reached.")
                    break

                try:
                    chunk = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                chunk = np.squeeze(chunk)
                level = self._compute_audio_level(chunk)
                print(f"[MicListener] level={level:.2f}")

                if level > self.speech_threshold:
                    if not is_speaking:
                        print(f"[MicListener] Speech started at level={level:.2f}")
                        is_speaking = True
                        speech_buffer = []

                    speech_buffer.append(chunk.copy())
                    silence_counter = 0

                elif is_speaking:
                    speech_buffer.append(chunk.copy())
                    silence_counter += 1

                    if silence_counter >= self.silence_limit:
                        print(f"[MicListener] Speech ended, chunks={len(speech_buffer)}")
                        break

        if len(speech_buffer) == 0:
            return None

        return np.concatenate(speech_buffer).astype(np.int16)
