# Version 1.0

import json
import os

PROFILE_PATH = "user_profile.json"

DEFAULT_PROFILE = {
    "name": None,
    "preferred_language": None,
    "preferred_answer_style": None
}


def load_user_profile():
    if not os.path.exists(PROFILE_PATH):
        return DEFAULT_PROFILE.copy()

    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile = DEFAULT_PROFILE.copy()
        profile.update(data)
        return profile

    except Exception:
        return DEFAULT_PROFILE.copy()


def save_user_profile(profile: dict):
    print("Saving profile to:", os.path.abspath(PROFILE_PATH))
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def update_user_profile_field(field: str, value: str):
    profile = load_user_profile()

    if field in profile:
        profile[field] = value.strip() if isinstance(value, str) else value
        save_user_profile(profile)

    return profile


def get_profile_summary_text(profile: dict) -> str:
    facts = []

    if profile.get("name"):
        facts.append(f"The user's name is {profile['name']}.")

    if profile.get("preferred_language"):
        facts.append(f"The user prefers responses in {profile['preferred_language']}.")

    if profile.get("preferred_answer_style"):
        facts.append(f"The user prefers {profile['preferred_answer_style']} answers.")

    return "\n".join(facts)


import re

def extract_profile_updates_from_text(text: str):
    updates = {}
    lower_text = text.lower().strip()

    # Strong patterns for explicit naming
    strong_name_patterns = [
        r"\bmy name is\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)",
        r"\bcall me\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)"
    ]

    for pattern in strong_name_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            extracted_name = match.group(1).strip()
            extracted_name = " ".join(part.capitalize() for part in extracted_name.split())
            updates["name"] = extracted_name
            break

    # Safer "I am ..." handling
    if "name" not in updates:
        match = re.search(r"\bi am\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b", text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"\bi'm\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b", text, flags=re.IGNORECASE)

        if match:
            candidate = match.group(1).strip()
            candidate_lower = candidate.lower()

            blocked_words = {
                "studying", "working", "doing", "trying", "using", "asking",
                "building", "looking", "going", "learning", "taking",
                "interested", "from", "in", "at", "on", "for", "thinking"
            }

            parts = candidate_lower.split()

            # Accept only short, name-like candidates
            if (
                len(parts) <= 2 and
                all(part not in blocked_words for part in parts) and
                all(len(part) >= 2 for part in parts)
            ):
                extracted_name = " ".join(part.capitalize() for part in candidate.split())
                updates["name"] = extracted_name

    # Preferred language
    if "answer in english" in lower_text or "respond in english" in lower_text:
        updates["preferred_language"] = "English"
    elif "answer in bangla" in lower_text or "respond in bangla" in lower_text:
        updates["preferred_language"] = "Bangla"

    # Preferred answer style
    if "concise" in lower_text or "short answers" in lower_text or "brief answers" in lower_text or "short response" in lower_text or "brief reponse" in lower_text:
        updates["preferred_answer_style"] = "concise"
    elif "detailed" in lower_text or "elaborate" in lower_text or "long answers" in lower_text or "broad" in lower_text or "long reponse" in lower_text or "broad response" in lower_text:
        updates["preferred_answer_style"] = "detailed"

    return updates

def apply_profile_updates_from_text(text: str):
    updates = extract_profile_updates_from_text(text)
    if not updates:
        return None

    profile = load_user_profile()
    changed = {}

    for field, value in updates.items():
        if profile.get(field) != value:
            profile[field] = value
            changed[field] = value

    if changed:
        save_user_profile(profile)

    return changed if changed else None
