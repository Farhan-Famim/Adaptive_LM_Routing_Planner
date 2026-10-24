import collections
import queue
import threading
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


class InterruptMonitor:
    """
    Listens to the microphone WHILE the assistant is speaking.

    If the user talks for a moment (sustained speech, not a click or a cough), it:
      1. silences the assistant (speaker.interrupt()),
      2. keeps recording until the user stops talking (including the first words,
         thanks to a short "pre-roll" memory),
      3. hands the recording back through finish().

    Usage:
        monitor.start(speaker)        # just before the assistant starts speaking
        ...                           # assistant speaks / the reply is displayed
        audio = monitor.finish()      # None if the user did not interrupt
    """

    def __init__(self, listener, trigger_threshold=None, sustain_chunks=6, pre_roll_chunks=8):
        self.listener = listener
        # Loudness the user must exceed to count as speaking (same scale as the listener).
        self.trigger_threshold = trigger_threshold if trigger_threshold is not None else listener.speech_threshold
        # How many loud chunks in a row = a real interruption (1 chunk = blocksize/sample_rate = 0.064 s).
        self.sustain_chunks = sustain_chunks
        # How many chunks before the trigger are kept, so the first words are not cut off.
        self.pre_roll_chunks = pre_roll_chunks

        self.triggered = threading.Event()   # the user interrupted
        self._stop = threading.Event()       # ask the thread to end
        self._done = threading.Event()       # recording finished (or thread ended)
        self._thread = None
        self._audio = None

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self, speaker):
        self._shutdown()   # make sure no older monitor is still running
        self.triggered.clear()
        self._stop.clear()
        self._done.clear()
        self._audio = None
        self._thread = threading.Thread(
            target=self._run, args=(speaker,), daemon=True, name="interrupt-monitor"
        )
        self._thread.start()

    def finish(self, timeout=None):
        """Stop monitoring. Returns the user's audio (int16 array) if they interrupted, else None."""
        if timeout is None:
            timeout = self.listener.max_record_seconds + 3
        if self.triggered.is_set():
            self._done.wait(timeout)   # let the user finish their sentence
        audio = self._audio
        self._shutdown()
        return audio

    def cancel(self):
        """Stop monitoring and throw away anything recorded (used when a reply is aborted)."""
        self._shutdown()
        self.triggered.clear()
        self._audio = None

    def _shutdown(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self, speaker):
        l = self.listener
        pre_roll = collections.deque(maxlen=self.pre_roll_chunks)
        loud_run = 0
        recording = False
        buffer = []
        silence = 0
        rec_start = 0.0

        try:
            with sd.InputStream(
                samplerate=l.sample_rate,
                channels=l.channels,
                blocksize=l.blocksize,
                dtype="int16",
            ) as stream:
                while not self._stop.is_set():
                    data, _ = stream.read(l.blocksize)
                    chunk = np.squeeze(data).copy()
                    level = l._compute_audio_level(chunk)

                    if not recording:
                        pre_roll.append(chunk)
                        loud_run = loud_run + 1 if level > self.trigger_threshold else 0
                        if loud_run >= self.sustain_chunks:
                            print(f"[InterruptMonitor] User interrupted (level={level:.0f})")
                            speaker.interrupt()
                            self.triggered.set()
                            buffer = list(pre_roll)
                            recording = True
                            silence = 0
                            rec_start = time.time()
                    else:
                        buffer.append(chunk)
                        silence = 0 if level > l.speech_threshold else silence + 1
                        if silence >= l.silence_limit or time.time() - rec_start > l.max_record_seconds:
                            self._audio = np.concatenate(buffer).astype(np.int16)
                            break
        except Exception as e:
            print(f"[InterruptMonitor] error: {e}")
        finally:
            self._done.set()