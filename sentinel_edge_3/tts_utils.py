import io
import os
import queue
import re
import threading
import time

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from tqdm import tqdm

# Define exact model files
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

MODELS_DIR = "tts_models"
MODEL_PATH = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(MODELS_DIR, "voices-v1.0.bin")

_kokoro_instance = None

# CPU threads Kokoro may use. Phi-3 (Ollama) also uses the CPU; if both grab every
# core they slow each other down. Your CPU has 8 physical cores: keep Kokoro + Phi-3
# (Ollama's num_thread) at 8 or fewer in total. Set to 0 to let ONNX Runtime decide.
TTS_THREADS = 4


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 Megabyte

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
        print("Downloading Kokoro TTS model (this is a one-time ~300MB download)...")
        download_file(MODEL_URL, MODEL_PATH)

    if not os.path.exists(VOICES_PATH):
        print("Downloading Kokoro TTS voices file...")
        download_file(VOICES_URL, VOICES_PATH)

    print("Loading Kokoro ONNX model...")
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = TTS_THREADS
        session = ort.InferenceSession(
            MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        _kokoro_instance = Kokoro.from_session(session, VOICES_PATH)
    except Exception as e:
        print(f"Custom ONNX session failed ({e}); using default Kokoro setup.")
        _kokoro_instance = Kokoro(MODEL_PATH, VOICES_PATH)
    print("Kokoro model loaded successfully!")
    return _kokoro_instance


# ======================================================================
# Original (non-streaming) helpers - kept for backward compatibility
# ======================================================================
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

        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
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


# ======================================================================
# Streaming TTS: speak the reply sentence-by-sentence while it is
# still being generated.
#
#   LLM chunks -> feed() -> sentence splitter -> [synth thread]
#              -> audio queue -> [playback thread] -> speakers
#
# Synthesis of sentence N+1 happens while sentence N is playing, so
# after the first sentence the speech is gapless.
# ======================================================================
_SENT_END = re.compile(r"(?<!\d)[.!?](?=\s)|\n")              # sentence end (not "1." list numbers)
_CLAUSE_END = re.compile(r"(?:(?<!\d)[.!?]|[,;:])(?=\s)|\n")  # sentence end OR clause break
_EMOJI = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27bf\ufe0f]")


def clean_for_speech(text: str) -> str:
    """Strip markdown/URLs/emoji so the TTS doesn't read symbols aloud."""
    text = re.sub(r"`([^`]*)`", r"\1", text)                 # inline code
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)     # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)                 # bare URLs
    text = re.sub(r"^\s*(?:[-*\u2022]|\d+\.)\s+", "", text, flags=re.M)  # list markers
    text = re.sub(r"[*_#>~|]+", "", text)                    # markdown symbols
    text = _EMOJI.sub("", text)

    # Join lines with a pause so list items don't run together
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
    out = ""
    for ln in filter(None, lines):
        out += ln if not out else (" " if out[-1] in ".!?:;," else ". ") + ln
    return out


def _boost_process_priority():
    """Windows only: run this app at 'above normal' priority. When Phi-3 (Ollama, a separate
    process) and speech synthesis compete for the CPU, the voice then wins and doesn't stall."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception as e:
        print(f"Could not raise process priority: {e}")


class StreamingSpeaker:
    MIN_CHARS = 12   # whole-sentence mode: shorter sentences are merged with the next one
    MAX_CHARS = 220  # force a split if no sentence end shows up by here

    # How chunks are cut (all sizes in characters):
    #  - nothing queued (start of reply / audio ran dry): speak a SMALL chunk at once
    #  - only one chunk queued: next chunk may be at most GROWTH x the previous one,
    #    because it is synthesized while the previous one plays
    #  - 2+ chunks already queued: wait for whole sentences (better intonation)
    BOOST_PRIORITY = True     # Windows: favour speech over Phi-3 when the CPU is busy
    DEBUG = False             # True = print timing info to the terminal (to find the cause of pauses)
    PLAYBACK_LATENCY = "low"  # try "high" if the log shows underflows / device stalls

    START_CHUNK = (18, 45)   # (min, max) size of a chunk when nothing is queued
    GROWTH = 1.2

    def __init__(self, voice="af_heart", speed=1.0):
        self.voice = voice
        self.speed = speed

        if self.BOOST_PRIORITY:
            _boost_process_priority()

        self._text_q = queue.Queue()               # (gen, sentence)
        self._audio_q = queue.Queue(maxsize=6)     # (gen, samples, sample_rate)
        self._raw = ""         # text that may still contain a partial ``` fence
        self._buffer = ""      # speakable text waiting for a sentence end
        self._in_code = False
        self._last_len = 0      # length of the most recently queued chunk
        self._gen = 0          # bumped by stop(); stale items are discarded
        self._pending = 0      # sentences queued but not yet fully played
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._warmed = False
        self._muted = False     # True after interrupt(): ignore the rest of this reply
        self._t0 = time.perf_counter()       # debug clock
        self._play_clock_end = 0.0           # debug: when the audio written so far will finish playing

        threading.Thread(target=self._synth_loop, daemon=True, name="tts-synth").start()
        threading.Thread(target=self._play_loop, daemon=True, name="tts-play").start()

    # ------------------------- public API -------------------------
    def warmup(self):
        """Load the model and run one tiny synthesis so the first real sentence is fast."""
        if self._warmed:
            return True
        kokoro = get_kokoro_model()
        if not kokoro:
            return False
        try:
            kokoro.create("Ready.", voice=self.voice, speed=self.speed, lang="en-us")
        except Exception as e:
            print(f"TTS warmup failed: {e}")
        self._warmed = True
        return True

    def start(self):
        """Call at the beginning of every new reply (also cuts off any old speech)."""
        self.stop()
        self._muted = False
        self._t0 = time.perf_counter()
        self._play_clock_end = 0.0

    def feed(self, text: str):
        """Feed streamed LLM text. Complete sentences are spoken immediately."""
        if not text or self._muted:
            return
        self._raw += text
        self._buffer += self._strip_code()
        self._flush_sentences()

    def finish(self):
        """Call once the LLM is done: speaks whatever is left in the buffer."""
        if self._muted:
            return
        self._flush_sentences()
        tail, self._buffer = self._buffer, ""
        self._raw = ""
        self._enqueue(tail)

    def wait_until_done(self, timeout=None) -> bool:
        """Block until everything queued has been played."""
        return self._idle.wait(timeout)

    def is_speaking(self) -> bool:
        return not self._idle.is_set()

    def stop(self):
        """Immediately silence and clear everything."""
        with self._lock:
            self._gen += 1
            self._raw = ""
            self._buffer = ""
            self._in_code = False
            self._last_len = 0
            self._pending = 0
            self._drain(self._text_q)
            self._drain(self._audio_q)
            self._idle.set()

    def interrupt(self):
        """The user started talking: silence the assistant now AND ignore the rest of this
        reply's text (the text still appears on screen, it just isn't spoken).
        The next start() lifts this."""
        self._muted = True
        self.stop()

    @property
    def was_interrupted(self) -> bool:
        return self._muted

    def speak(self, text: str, block: bool = True):
        """Convenience for a complete string (e.g. fixed replies)."""
        self.start()
        self.feed(text)
        self.finish()
        if block:
            self.wait_until_done(timeout=120)

    # ------------------------- internals -------------------------
    def _log(self, msg):
        if self.DEBUG:
            print(f"[TTS {time.perf_counter() - self._t0:6.2f}s] {msg}", flush=True)

    @staticmethod
    def _drain(q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _strip_code(self) -> str:
        """Remove fenced ``` code blocks from self._raw (fences may arrive split
        across chunks) and return the speakable text. A trailing partial fence
        (1-2 backticks) is held back in self._raw until the next chunk."""
        out, buf = [], self._raw
        while True:
            idx = buf.find("```")
            if idx == -1:
                m = re.search(r"`{1,2}$", buf)
                held = m.group(0) if m else ""
                if not self._in_code:
                    out.append(buf[: len(buf) - len(held)])
                buf = held
                break
            if not self._in_code:
                out.append(buf[:idx])
            buf = buf[idx + 3:]
            self._in_code = not self._in_code
        self._raw = buf
        return "".join(out)

    def _next_cut(self) -> int:
        """Where the next chunk should end in self._buffer (0 = keep waiting)."""
        buf = self._buffer
        pending = self._pending          # chunks queued but not fully played yet

        if pending == 0:                 # reply just started, or the audio ran dry
            lo, hi = self.START_CHUNK
            pattern = _CLAUSE_END
        elif pending == 1:               # thin buffer: grow gently
            lo = max(self.START_CHUNK[0], int(self._last_len * 0.7))
            hi = min(self.MAX_CHARS, max(self.START_CHUNK[1], int(self._last_len * self.GROWTH)))
            pattern = _CLAUSE_END
        else:                            # comfortable buffer: whole sentences
            lo, hi = self.MIN_CHARS, self.MAX_CHARS
            pattern = _SENT_END

        for m in pattern.finditer(buf):
            if m.end() > hi:
                break
            if m.end() >= lo:
                return m.end()

        if len(buf) >= hi:               # no good boundary in range: cut at a space
            cut = buf.rfind(" ", lo, hi)
            return cut if cut > 0 else hi
        return 0

    def _flush_sentences(self):
        while True:
            cut = self._next_cut()
            if not cut:
                break
            self._enqueue(self._buffer[:cut])
            self._buffer = self._buffer[cut:]

    def _enqueue(self, text: str):
        text = clean_for_speech(text)
        if not re.search(r"\w", text):
            return
        with self._lock:
            if self._muted:
                return
            self._last_len = len(text)
            self._pending += 1
            self._idle.clear()
            self._text_q.put((self._gen, text))
        self._log(f"text ready: queued chunk of {len(text)} chars (chunks waiting: {self._pending})")

    def _mark_done(self, gen, drain_delay=0.0):
        with self._lock:
            if gen != self._gen:
                return
            self._pending -= 1
            last = self._pending <= 0
        if last:
            if drain_delay:
                time.sleep(drain_delay)  # let the sound card empty its buffer
            with self._lock:
                if gen == self._gen and self._pending <= 0:
                    self._idle.set()

    def _synth_loop(self):
        while True:
            gen, text = self._text_q.get()
            if gen != self._gen:
                continue
            samples, sr = None, None
            kokoro = get_kokoro_model()
            if kokoro:
                try:
                    t_synth = time.perf_counter()
                    samples, sr = kokoro.create(
                        text, voice=self.voice, speed=self.speed, lang="en-us"
                    )
                    self._log(
                        f"synthesized {len(text)} chars in {time.perf_counter() - t_synth:.2f}s "
                        f"-> {len(samples) / sr:.1f}s of audio"
                    )
                except Exception as e:
                    print(f"TTS synthesis error: {e}")
            if samples is None:
                self._mark_done(gen)
                continue
            self._audio_q.put((gen, samples, sr))

    def _play_loop(self):
        stream, stream_sr = None, None

        def close_stream(abort=False):
            nonlocal stream, stream_sr
            if stream is not None:
                try:
                    (stream.abort if abort else stream.stop)()
                    stream.close()
                except Exception:
                    pass
            stream, stream_sr = None, None

        while True:
            try:
                gen, samples, sr = self._audio_q.get(timeout=2.0)
            except queue.Empty:
                close_stream()   # nothing to play: release the audio device
                continue

            if gen != self._gen:
                close_stream(abort=True)
                continue

            try:
                if stream is None or stream_sr != sr:
                    close_stream()
                    stream = sd.OutputStream(
                        samplerate=sr, channels=1, dtype="float32", latency=self.PLAYBACK_LATENCY
                    )
                    stream.start()
                    stream_sr = sr
                data = np.asarray(samples, dtype="float32").reshape(-1, 1)

                if self.DEBUG:
                    now = time.perf_counter()
                    if self._play_clock_end and now > self._play_clock_end + 0.15:
                        self._log(
                            f"*** SILENCE: audio ran dry for {now - self._play_clock_end:.2f}s "
                            f"(next chunk was not ready; synthesis queue={self._text_q.qsize()})"
                        )
                    self._play_clock_end = max(now, self._play_clock_end) + len(data) / sr

                for i in range(0, len(data), 4096):   # small blocks so stop() reacts fast
                    if gen != self._gen:
                        break
                    block = data[i:i + 4096]
                    t_write = time.perf_counter()
                    underflow = stream.write(block)
                    if self.DEBUG:
                        took = time.perf_counter() - t_write
                        if underflow:
                            self._log("*** audio underflow (sound device ran out of data)")
                        if took > 2 * len(block) / sr + 0.3:
                            self._log(f"*** device stall: one audio write took {took:.2f}s")
            except Exception as e:
                print(f"TTS playback error: {e}")
                close_stream(abort=True)

            if gen != self._gen:
                close_stream(abort=True)
                continue

            latency = getattr(stream, "latency", 0.0) if stream is not None else 0.0
            self._mark_done(gen, drain_delay=float(latency) if isinstance(latency, (int, float)) else 0.0)


_speaker = None


def get_streaming_speaker() -> StreamingSpeaker:
    """Process-wide singleton, so it survives Streamlit reruns."""
    global _speaker
    if _speaker is None:
        _speaker = StreamingSpeaker()
    return _speaker