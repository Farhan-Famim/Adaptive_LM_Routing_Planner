# Version 7.0
# Can remember chat context
# Supports multiple chats

# Note: In streamlit settings, trun on/off TTS first, before Auto Voice mode.

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import uuid
import json
import requests
import re
import time

from audio_utils import load_whisper_model, transcribe_audio_file
from tts_utils import generate_speech_bytes, play_speech_bytes

from mic_listener import MicUtteranceListener

from profile_memory import load_user_profile, get_profile_summary_text, apply_profile_updates_from_text
from chat_store import list_chats, load_chat, save_chat, new_chat, delete_chat

# ===============================
# CONFIG
# ===============================
LOCAL_MODEL = "phi3:3.8b"
OPENROUTER_MODEL = "google/gemini-3-flash-preview"
MEMORY_PATH = "./memory_db"
OPENROUTER_API_KEY = "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"

# ===============================
# CLAWROUTER CONFIG
# ===============================
DIMENSION_WEIGHTS = {
    "tokenCount": 0.08,
    "codePresence": 0.15,
    "reasoningMarkers": 0.18,
    "technicalTerms": 0.10,
    "creativeMarkers": 0.05,
    "simpleIndicators": 0.02,
    "multiStepPatterns": 0.12,
    "questionComplexity": 0.05,
    "imperativeVerbs": 0.03,
    "constraintCount": 0.04,
    "outputFormat": 0.03,
    "referenceComplexity": 0.02,
    "negationComplexity": 0.01,
    "domainSpecificity": 0.02,
    "agenticTask": 0.04,
    "timeSensitivity": 0.40,
}

KEYWORDS = {
    "codePresence": ["function", "class", "import", "def", "SELECT", "async", "await", "const", "let", "var", "return", "```"],
    "reasoningMarkers": ["prove", "theorem", "derive", "step by step", "chain of thought", "formally", "mathematical", "proof", "logically"],
    "simpleIndicators": ["what is", "define", "translate", "hello", "yes or no", "capital of", "how old", "who is", "when was"],
    "technicalTerms": ["algorithm", "optimize", "architecture", "distributed", "kubernetes", "microservice", "database", "infrastructure"],
    "creativeMarkers": ["story", "poem", "compose", "brainstorm", "creative", "imagine", "write a"],
    "imperativeVerbs": ["build", "create", "implement", "design", "develop", "construct", "generate", "deploy", "configure", "set up"],
    "constraintCount": ["under", "at most", "at least", "within", "no more than", "o(", "maximum", "minimum", "limit", "budget"],
    "outputFormat": ["json", "yaml", "xml", "table", "csv", "markdown", "schema", "format as", "structured"],
    "referenceComplexity": ["above", "below", "previous", "following", "the docs", "the api", "the code", "earlier", "attached"],
    "negationComplexity": ["don't", "do not", "avoid", "never", "without", "except", "exclude", "no longer"],
    "domainSpecificity": ["quantum", "fpga", "vlsi", "risc-v", "asic", "photonics", "genomics", "proteomics", "topological", "homomorphic", "zero-knowledge", "lattice-based"],
    "agenticTask": ["read file", "read the file", "look at", "check the", "open the", "edit", "modify", "update the", "change the", "write to", "create file", "execute", "deploy", "install", "npm", "pip", "compile", "after that", "and also", "once done", "step 1", "step 2", "fix", "debug", "until it works", "keep trying", "iterate", "make sure", "verify", "confirm"],
    "timeSensitivity": [
        "today", "tomorrow", "yesterday", "this week", "last week", "next week",
        "this month", "last month", "next month", "this year", "last year", "next year",
        "currently", "now", "as of", "recent", "latest", "upcoming", "ongoing", "current",
        "breaking", "news", "score", "result", "update", "announcement", 
        "weather", "forecast", "temperature", "alert", "status", "rate", "price", 
        "exchange", "market", "stock", "trend", "poll", "ranking", "ranking update", 
        "event", "deadline", "schedule", "availability",
        "ceo", "president", "prime minister", "manager", "chairman", "leader", 
        "governor", "director", "head of",
        "interest rate", "value", "exchange rate", "budget", "funding", 
        "loan rate", "tax rate", "inflation", "crypto", "bitcoin", "ethereum",
        "match", "game", "fixture", "league table", "tournament", "final", "semi-final", "qualification", "draw",
        "change", "recently", "new", "latest version", "breaking news",
        "happening now", "current status", "as of now", "real-time"
    ]
}

# ===============================
# SYSTEM PROMPTS
# ===============================
SYSTEM_PROMPT = """
You are a highly capable, efficient, and honest AI assistant.

CRITICAL INSTRUCTIONS:
1. For simple questions or fact retrieval, respond as concisely and directly as possible without unnecessary conversational filler. However, if a prompt clearly requires an explanation, provide a detailed and comprehensive answer.
2. DO NOT HALLUCINATE. Never make up facts, names, or details. If you aren't absolutely sure, just say "I don't know."
3. If the user simply greets you (e.g., "hello", "hi", "how are you"), just respond with a normal greeting. Do not offer additional facts.
4. When context facts are provided, use them to answer. The user has voluntarily shared this personal information with you — it is completely safe and appropriate to repeat it back. For example, if the user told you their name, you must answer "What is my name?" directly using that name. Never refuse to answer a personal question if the answer exists in the provided context. But NEVER output the exact phrase "<context_from_memory>" or use words like "context", "memory", "database", or "previous conversation" in your responses. If the provided context does not contain the answer to a specific personal question, simply state that you don't know.
5. NEVER refuse to answer a question by citing privacy, safety, or ethical concerns when the answer is simply a personal fact the user themselves told you (like their name, country, age, job, preferences). These are not sensitive requests — they are the user asking you to recall what they said. Treat them as normal factual questions.
6. If the user's question is completely unrelated to the provided context, completely IGNORE the context entirely and answer normally.
"""


# ===============================
# INIT RAG / MEMORY
# ===============================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=MEMORY_PATH)
    return client.get_or_create_collection(
        "memory",
        metadata={"hnsw:space": "cosine"}
    )

@st.cache_resource
def get_mic_listener():
    return MicUtteranceListener()

@st.cache_resource
def get_whisper_model():
    return load_whisper_model()

@st.cache_resource
def get_user_profile():
    return load_user_profile()

embedder = load_embedder()
collection = load_collection()
whisper_model = get_whisper_model()
mic_listener = get_mic_listener()
user_profile = get_user_profile()

@st.fragment(run_every="1s")
def auto_voice_listener():
    if not st.session_state.get("auto_voice_mode_active", False):
        return

    if st.session_state.get("pending_voice_input") is not None:
        return
    
    if st.session_state.get("tts_playing", False):
        st.session_state["voice_status"] = "Assistant is speaking..."
        return

    if time.time() < st.session_state.get("resume_listening_at", 0.0):
        st.session_state["voice_status"] = "Waiting before listening..."
        return

    st.session_state["voice_status"] = "Listening from microphone..."
    audio_array = mic_listener.capture_single_utterance()

    if audio_array is not None and len(audio_array) > 0:
        from speech_capture_utils import pcm_to_wav_bytes, InMemoryAudioFile

        st.session_state["voice_status"] = "Transcribing speech..."
        wav_bytes = pcm_to_wav_bytes(audio_array, sample_rate=16000)
        audio_file = InMemoryAudioFile(wav_bytes)

        transcribed_text = transcribe_audio_file(audio_file, whisper_model)

        if transcribed_text and not transcribed_text.startswith("__ERROR__:"):
            st.session_state["pending_voice_input"] = transcribed_text
            st.session_state["input_mode"] = "voice"
            st.session_state["voice_status"] = f"Recognized: {transcribed_text}"
            st.rerun()
        else:
            st.session_state["voice_status"] = "Speech captured, but transcription failed."

# ===============================
# CLAWROUTER LOGIC
# ===============================
def score_token_count(text):
    tokens = len(text.split())
    if tokens < 50: return -1.0
    if tokens > 500: return 1.0
    return 0.0

def score_keywords(text, kw_list, thresholds, scores):
    matches = sum(1 for kw in kw_list if re.search(r'(?<!\w)' + re.escape(kw.lower()) + r'(?!\w)', text))
    if matches >= thresholds['high']: return scores['high'], matches
    if matches >= thresholds['low']: return scores['low'], matches
    return scores['none'], matches

def score_multistep(text):
    if re.search(r'first.*then|step \d|\d\.\s', text, re.IGNORECASE):
        return 0.5
    return 0.0

def score_question_complexity(text):
    if text.count('?') > 3:
        return 0.5
    return 0.0

def classify_prompt_tier(prompt: str):
    text = prompt.lower()
    dimensions = {}
    
    dimensions['tokenCount'] = score_token_count(text)
    dimensions['codePresence'], _ = score_keywords(text, KEYWORDS['codePresence'], {'low': 1, 'high': 2}, {'none': 0, 'low': 0.5, 'high': 1.0})
    dimensions['reasoningMarkers'], reasoning_matches = score_keywords(text, KEYWORDS['reasoningMarkers'], {'low': 1, 'high': 2}, {'none': 0, 'low': 0.7, 'high': 1.0})
    dimensions['technicalTerms'], _ = score_keywords(text, KEYWORDS['technicalTerms'], {'low': 2, 'high': 4}, {'none': 0, 'low': 0.5, 'high': 1.0})
    dimensions['creativeMarkers'], _ = score_keywords(text, KEYWORDS['creativeMarkers'], {'low': 1, 'high': 2}, {'none': 0, 'low': 0.5, 'high': 0.7})
    dimensions['simpleIndicators'], _ = score_keywords(text, KEYWORDS['simpleIndicators'], {'low': 1, 'high': 2}, {'none': 0, 'low': -1.0, 'high': -1.0})
    dimensions['multiStepPatterns'] = score_multistep(text)
    dimensions['questionComplexity'] = score_question_complexity(text)
    dimensions['imperativeVerbs'], _ = score_keywords(text, KEYWORDS['imperativeVerbs'], {'low': 1, 'high': 2}, {'none': 0, 'low': 0.3, 'high': 0.5})
    dimensions['constraintCount'], _ = score_keywords(text, KEYWORDS['constraintCount'], {'low': 1, 'high': 3}, {'none': 0, 'low': 0.3, 'high': 0.7})
    dimensions['outputFormat'], _ = score_keywords(text, KEYWORDS['outputFormat'], {'low': 1, 'high': 2}, {'none': 0, 'low': 0.4, 'high': 0.7})
    dimensions['referenceComplexity'], _ = score_keywords(text, KEYWORDS['referenceComplexity'], {'low': 1, 'high': 2}, {'none': 0, 'low': 0.3, 'high': 0.5})
    dimensions['negationComplexity'], _ = score_keywords(text, KEYWORDS['negationComplexity'], {'low': 2, 'high': 3}, {'none': 0, 'low': 0.3, 'high': 0.5})
    dimensions['domainSpecificity'], _ = score_keywords(text, KEYWORDS['domainSpecificity'], {'low': 1, 'high': 2}, {'none': 0, 'low': 0.5, 'high': 0.8})
    dimensions['timeSensitivity'], time_matches = score_keywords(text, KEYWORDS['timeSensitivity'], {'low': 1, 'high': 2}, {'none': 0, 'low': 0.6, 'high': 1.0})
    
    agentic_matches = sum(1 for kw in KEYWORDS['agenticTask'] if re.search(r'(?<!\w)' + re.escape(kw.lower()) + r'(?!\w)', text))
    if agentic_matches >= 4:
        dimensions['agenticTask'] = 1.0
    elif agentic_matches >= 3:
        dimensions['agenticTask'] = 0.6
    elif agentic_matches >= 1:
        dimensions['agenticTask'] = 0.2
    else:
        dimensions['agenticTask'] = 0.0

    # Calculate weighted score
    weighted_score = sum(dimensions[dim] * DIMENSION_WEIGHTS[dim] for dim in dimensions)
    
    tier = ""
    # Direct overrides
    if time_matches >= 1:
        tier = "TIME_SENSITIVE"
    elif reasoning_matches >= 2:
        tier = "REASONING"
    else:
        if weighted_score < 0.0:
            tier = "SIMPLE"
        elif weighted_score < 0.3:
            tier = "MEDIUM"
        elif weighted_score < 0.5:
            tier = "COMPLEX"
        else:
            tier = "REASONING"
            
    active_dimensions = {k: v for k, v in dimensions.items() if v != 0.0}
            
    return tier, weighted_score, active_dimensions

# ===============================
# RAG LOGIC (MEMORY)
# ===============================
def remember(text: str):
    # Rewrite the memory into an objective fact using the local LLM
    prompt = f"""
Rewrite the following information into a single clear, objective, third-person factual statement.
If the text uses 'I', 'me', 'my', it refers to 'the user'.
Do not include any conversational filler like "Here is the rewritten fact", just output the fact itself.

Original text: {text}
"""
    try:
        response = ollama.chat(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        fact = response["message"]["content"].strip()
        # Clean up any quotes the LLM might have added
        if fact.startswith('"') and fact.endswith('"'):
            fact = fact[1:-1]
    except Exception as e:
        fact = text # Fallback to original text if LLM fails
        
    embedding = embedder.encode(fact).tolist()

    existing_count = collection.count()

    if existing_count > 0:
        check_k = min(3, existing_count)
        existing = collection.query(
            query_embeddings=[embedding],
            n_results=check_k
        )

        existing_docs = existing.get("documents", [[]])[0]
        existing_distances = existing.get("distances", [[]])[0]

        for doc, dist in zip(existing_docs, existing_distances):
            # very similar memory already exists
            if dist < 0.12:
                return doc

    collection.add(
        documents=[fact],
        embeddings=[embedding],
        ids=[str(uuid.uuid4())]
    )
    return fact

def recall(query: str, k: int = 4):
    if collection.count() == 0:
        return []

    search_query = f"Represent this sentence for searching relevant passages: {query}"
    embedding = embedder.encode(search_query).tolist()
    
    actual_k = min(k, collection.count())
    results = collection.query(
        query_embeddings=[embedding], 
        n_results=actual_k
    )
    
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    valid_memories = []
    # Distance Threshold: Only keep relevant memories (BAAI BGE cosine distance)
    for doc, dist in zip(docs, distances):
        # 0.65 is a robust cutoff for BGE-small cosine distance to allow query variations
        if dist < 0.65:
            valid_memories.append(doc)
            
    return valid_memories


def should_store_semantic_memory(text: str) -> bool:
    lower_text = text.lower().strip()

    # These should go to structured profile memory, not semantic memory
    profile_like_patterns = [
        "my name is",
        "i am ",
        "i'm ",
        "call me",
        "answer in english",
        "respond in english",
        "respond in bangla",
        "respond in bengali",
        "concise",
        "broad",
        "detailed",
        "long answers",
        "elaborate",
        "i prefer",
    ]

    if any(pattern in lower_text for pattern in profile_like_patterns):
        return False

    memory_triggers = [
        "remember:",
        "i am working on",
        "i'm working on",
        "my project is",
        "i am building",
        "i'm building",
        "i usually use",
        "i use",
        "for future reference",
        "keep in mind that",
    ]

    return any(trigger in lower_text for trigger in memory_triggers)


def auto_store_semantic_memory(text: str):
    if not should_store_semantic_memory(text):
        return None

    # Avoid storing short/unhelpful text
    if len(text.split()) < 4:
        return None

    stored_fact = remember(text)
    return stored_fact


# ===============================
# CHAT CONTEXT BUILDER
# ===============================
CONTEXT_WINDOW = 2   # last N messages sent to LLM (keep small for speed)
SUMMARIZE_EVERY = 4  # summarize older messages every N new messages
SUMMARIZE_AFTER = 5 # only start summarizing after this many messages

def build_llm_history(messages: list) -> list:
    """Return only the last CONTEXT_WINDOW messages for the LLM."""
    return messages[-CONTEXT_WINDOW:]

def should_summarize(messages: list) -> bool:
    n = len(messages)
    return n >= SUMMARIZE_AFTER and n % SUMMARIZE_EVERY == 0

def generate_summary(messages: list, existing_summary: str) -> str:
    """
    Compress older messages into a brief summary using the local LLM.
    Only called occasionally (every SUMMARIZE_EVERY turns after threshold).
    """
    older = messages[:-CONTEXT_WINDOW]  # everything except recent window
    if not older:
        return existing_summary

    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"  # cap each msg at 300 chars
        for m in older
    )

    prompt = f"""You are summarizing a conversation to preserve key context.

Previous summary (if any): {existing_summary or 'None'}

New conversation to add:
{convo_text}

Write a NEW concise summary (max 6 lines) that captures: topics discussed, key facts the user mentioned, and any decisions or preferences. Be objective, third-person, dense.
Output ONLY the summary text, no preamble."""

    try:
        response = ollama.chat(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response["message"]["content"].strip()
    except:
        return existing_summary  # fallback: keep old summary if LLM fails

# ===============================
# LLM HANDLers
# ===============================
def ask_local_llm(history: list, memory_text: str = None, profile_text: str = None, summary_text: str = None):
    system_prompt = SYSTEM_PROMPT

    if profile_text:
        system_prompt += f"\n\n<context_from_profile>\n{profile_text}\n</context_from_profile>"

    if summary_text:
        system_prompt += f"\n\n<context_from_chat>\n{summary_text}\n</context_from_chat>"

    if memory_text:
        system_prompt += f"\n\n<context_from_memory>\n{memory_text}\n</context_from_memory>"
            
    messages = [{"role": "system", "content": system_prompt}] + history
    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=messages,
        stream=False,
    )
    answer = response["message"]["content"]
    
    def stream_local(s):
        import time
        for i in range(0, len(s), 5):
            yield s[i:i+5]
            time.sleep(0.01)
            
    return stream_local(answer)

def ask_openrouter(question: str, history: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True
        }),
        stream=True
    )

    def stream_generator():
        full_text = ""
        if response.status_code != 200:
            error_data = response.text
            yield f"⚠️ API Error ({response.status_code}): {error_data}"
            return

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        yield content
                except:
                    continue

        if not full_text:
            yield "⚠️ No response received from cloud model."

    return stream_generator()

# ===============================
# ROUTING
# ===============================
def route_question(question: str, history: list, summary: str = ""):

    profile = load_user_profile()
    profile_text = get_profile_summary_text(profile)

    if not profile_text.strip():
        profile_text = None

    # 1. Perform 15-Dimension scoring FIRST
    tier, score, active_dims = classify_prompt_tier(question)
    
    dims_data = {
        "tier": tier,
        "score": score,
        "active_dims": active_dims
    }

    # 2. Decision - Route to Cloud LLM if appropriate
    trimmed_history = build_llm_history(history)
    if tier in ["TIME_SENSITIVE", "COMPLEX", "REASONING"]:
        return ask_openrouter(question, trimmed_history), "OPENROUTER", dims_data

    # 3. If routed to Local LLM, check Memory (RAG)
    memories = recall(question)
    memory_text = "\n".join(memories)

    if memory_text:
        dims_data["tier"] = f"MEMORY_MATCH ({tier})"
        dims_data["active_dims"]["memory_injection"] = 1.0
        
        # Direct route to local LLM with cleanly separated memory context
        return ask_local_llm(trimmed_history, memory_text=memory_text, profile_text=profile_text, summary_text=summary or None), "LOCAL", dims_data

    # 4. Fallback: ask local LLM without memory
    return ask_local_llm(trimmed_history, profile_text=profile_text, summary_text=summary or None), "LOCAL", dims_data

# ===============================
# UI
# ===============================
st.set_page_config(page_title="LLM Chat", page_icon="🤖", layout="wide")

# ---- Sidebar: Chat Sessions + Settings ----
with st.sidebar:
    st.title("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        chat = new_chat()
        st.session_state["current_chat_id"] = chat["id"]
        st.session_state["messages"] = []
        st.session_state["chat_summary"] = ""
        st.rerun()

    st.divider()

    all_chats = list_chats()
    for c in all_chats:
        col1, col2 = st.columns([5, 1])
        is_active = c["id"] == st.session_state.get("current_chat_id")
        label = ("**" + c["title"] + "**") if is_active else c["title"]
        if col1.button(label, key=f"chat_{c['id']}", use_container_width=True):
            loaded = load_chat(c["id"])
            st.session_state["current_chat_id"] = c["id"]
            st.session_state["messages"] = loaded.get("messages", [])
            st.session_state["chat_summary"] = loaded.get("summary", "")
            st.rerun()
        if col2.button("🗑", key=f"del_{c['id']}"):
            delete_chat(c["id"])
            if is_active:
                st.session_state["current_chat_id"] = None
                st.session_state["messages"] = []
                st.session_state["chat_summary"] = ""
            st.rerun()

    st.divider()
    st.subheader("⚙️ Settings")
    tts_enabled = st.checkbox("Enable TTS (Kokoro)", value=False)
    auto_voice_mode = st.checkbox("Enable Auto Voice Mode", value=False)

# Custom CSS for a ChatGPT-like minimal UI
st.markdown("""
<style>
    /* Hide Streamlit header, footer, and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Adjust main container padding and max-width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 850px;
    }

    /* Style the chat messages */
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Sentinent Edge")

# --- Initialize or load current chat session ---
if "current_chat_id" not in st.session_state or st.session_state["current_chat_id"] is None:
    existing = list_chats()
    if existing:
        loaded = load_chat(existing[0]["id"])
        st.session_state["current_chat_id"] = existing[0]["id"]
        st.session_state["messages"] = loaded.get("messages", [])
        st.session_state["chat_summary"] = loaded.get("summary", "")
    else:
        chat = new_chat()
        st.session_state["current_chat_id"] = chat["id"]
        st.session_state["messages"] = []
        st.session_state["chat_summary"] = ""

if "chat_summary" not in st.session_state:
    st.session_state["chat_summary"] = ""

if "input_mode" not in st.session_state:
    st.session_state["input_mode"] = "text"

if "voice_status" not in st.session_state:
    st.session_state["voice_status"] = "Voice mode standby"

if "pending_voice_input" not in st.session_state:
    st.session_state["pending_voice_input"] = None

if "auto_voice_mode_active" not in st.session_state:
    st.session_state["auto_voice_mode_active"] = False

if "tts_playing" not in st.session_state:
    st.session_state["tts_playing"] = False

if "resume_listening_at" not in st.session_state:
    st.session_state["resume_listening_at"] = 0.0

st.session_state["auto_voice_mode_active"] = auto_voice_mode

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


st.markdown("### Input")
voice_status_placeholder = st.empty()
voice_status_placeholder.info(st.session_state["voice_status"])
typed_input = st.chat_input("Type a message...")
listen_once = st.button("🎤 Listen from microphone")

auto_voice_listener()

if listen_once and st.session_state["pending_voice_input"] is None:
    st.session_state["voice_status"] = "Listening from microphone..."
    voice_status_placeholder.info(st.session_state["voice_status"])

    audio_array = mic_listener.capture_single_utterance()

    if audio_array is not None and len(audio_array) > 0:
        from speech_capture_utils import pcm_to_wav_bytes, InMemoryAudioFile

        st.session_state["voice_status"] = "Transcribing speech..."
        voice_status_placeholder.info(st.session_state["voice_status"])

        wav_bytes = pcm_to_wav_bytes(audio_array, sample_rate=16000)
        audio_file = InMemoryAudioFile(wav_bytes)

        transcribed_text = transcribe_audio_file(audio_file, whisper_model)

        if transcribed_text and not transcribed_text.startswith("__ERROR__:"):
            st.session_state["pending_voice_input"] = transcribed_text
            st.session_state["input_mode"] = "voice"
            st.session_state["voice_status"] = f"Recognized: {transcribed_text}"
            voice_status_placeholder.success(st.session_state["voice_status"])
        else:
            st.session_state["voice_status"] = "Speech captured, but transcription failed."
            voice_status_placeholder.warning(st.session_state["voice_status"])
    else:
        st.session_state["voice_status"] = "No speech detected."
        voice_status_placeholder.warning(st.session_state["voice_status"])

user_input = None

if typed_input:
    st.session_state["input_mode"] = "text"
    user_input = typed_input

elif st.session_state["pending_voice_input"] is not None:
    user_input = st.session_state["pending_voice_input"]
    st.session_state["pending_voice_input"] = None

if user_input:
    apply_profile_updates_from_text(user_input)
    auto_store_semantic_memory(user_input)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar="👤"):
        if st.session_state.get("input_mode") == "voice":
            st.caption("🎤 Voice input")
        st.markdown(user_input)

    if user_input.lower().startswith("remember:"):
        memory_text = user_input[len("remember:"):].strip()
        if memory_text:
            fact_saved = remember(memory_text)
            assistant_reply = f"🧠 Memory stored: *{fact_saved}*"
        else:
            assistant_reply = "⚠️ Nothing to remember."

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(assistant_reply)
            if tts_enabled:
                st.session_state["tts_playing"] = True
                st.session_state["voice_status"] = "Assistant is speaking..."

                with st.spinner("Generating speech..."):
                    audio_bytes = generate_speech_bytes(assistant_reply)
                    if audio_bytes:
                        play_speech_bytes(audio_bytes)

                st.session_state["tts_playing"] = False
                st.session_state["resume_listening_at"] = time.time() + 1
                st.session_state["voice_status"] = "Voice mode standby"
    else:
        with st.chat_message("assistant", avatar="🤖"):

            stream, model_used, dims_data = route_question(
                user_input,
                st.session_state.messages,
                summary=st.session_state.get("chat_summary", "")
            )

            full_answer = ""
            placeholder = st.empty()

            for chunk in stream:
                full_answer += chunk
                placeholder.markdown(full_answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer}
        )

        # --- Auto-summarize older context (runs only occasionally) ---
        if should_summarize(st.session_state.messages):
            st.session_state["chat_summary"] = generate_summary(
                st.session_state.messages,
                st.session_state.get("chat_summary", "")
            )

        # --- Auto-title: use first user message as chat title ---
        chat_data = load_chat(st.session_state["current_chat_id"])
        if chat_data and chat_data["title"] == "New Chat":
            first_user_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), None)
            if first_user_msg:
                chat_data["title"] = first_user_msg[:40]

        # --- Save chat to disk ---
        if chat_data:
            chat_data["messages"] = st.session_state.messages
            chat_data["summary"] = st.session_state.get("chat_summary", "")
            save_chat(chat_data)
        
        if tts_enabled:
            st.session_state["tts_playing"] = True
            st.session_state["voice_status"] = "Assistant is speaking..."

            with st.spinner("Generating speech..."):
                audio_bytes = generate_speech_bytes(full_answer)
                if audio_bytes:
                    play_speech_bytes(audio_bytes)

            st.session_state["tts_playing"] = False
            st.session_state["resume_listening_at"] = time.time() + 1
            st.session_state["voice_status"] = "Voice mode standby"

        st.session_state["input_mode"] = "text"

        with st.expander("🔎 Router Diagnostics"):
            st.write(f"**Model Used:** {model_used}")
            st.write(f"**Final Tier Decision:** {dims_data['tier']}")
            st.write(f"**Weighted Prompt Score:** {dims_data['score']:.3f} [-0.3 to 0.5+]")
            if dims_data['active_dims']:
                st.write("**Triggered Logic Dimensions:**")
                st.json(dims_data['active_dims'])

