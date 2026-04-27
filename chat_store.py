# Version 1.0

import os, json, uuid, time

CHATS_DIR = "./chats"

def _ensure_dir():
    os.makedirs(CHATS_DIR, exist_ok=True)

def list_chats():
    """Return all chats sorted by newest first."""
    _ensure_dir()
    chats = []
    for f in os.listdir(CHATS_DIR):
        if f.endswith(".json"):
            with open(os.path.join(CHATS_DIR, f)) as fp:
                data = json.load(fp)
                chats.append({
                    "id": data["id"],
                    "title": data["title"],
                    "created_at": data["created_at"]
                })
    return sorted(chats, key=lambda x: x["created_at"], reverse=True)

def load_chat(chat_id: str):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_chat(chat: dict):
    _ensure_dir()
    with open(os.path.join(CHATS_DIR, f"{chat['id']}.json"), "w") as f:
        json.dump(chat, f)

def new_chat() -> dict:
    chat = {
        "id": str(uuid.uuid4())[:8],
        "title": "New Chat",
        "messages": [],
        "summary": "",          # compressed context of older messages
        "created_at": time.time()
    }
    save_chat(chat)
    return chat

def delete_chat(chat_id: str):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)
