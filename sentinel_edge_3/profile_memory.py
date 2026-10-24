# Version 1.1

import json
import os
import re

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
    # Write to a temp file first, then swap it in. If the app is closed in the middle
    # of a save, the old profile stays intact instead of becoming a broken file.
    tmp_path = PROFILE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, PROFILE_PATH)


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

    # "concise" is already the default behaviour of the system prompt, so don't spend
    # prompt tokens repeating it. Only mention styles that change the default.
    style = profile.get("preferred_answer_style")
    if style and style != "concise":
        facts.append(f"The user has asked for {style} answers.")

    return "\n".join(facts)


# ======================================================================
# Extracting profile information from what the user says
# ======================================================================

# A captured name is cut off at the first of these words, and a name can't start with one.
# (Stops "My name is Sam and I like..." -> "Sam", "Call me back" -> nothing, "I'm fine" -> nothing)
_NAME_STOP_WORDS = {
    # connectors / grammar
    "and", "but", "so", "or", "because", "who", "which", "that", "from", "in", "at", "on",
    "for", "to", "with", "is", "am", "are", "was", "i", "im", "my", "me", "the", "a", "an",
    "please", "can", "could", "would", "what", "how", "if", "when", "then", "too", "also",
    # time / "call me ..." endings
    "back", "later", "tomorrow", "tonight", "today", "now", "soon", "again", "anytime", "maybe",
    # common "I'm ..." replies that are not names
    "just", "not", "very", "really", "fine", "good", "great", "well", "okay", "ok", "sure",
    "sorry", "ready", "done", "here", "there", "glad", "happy", "sad", "tired", "hungry", "busy",
    # activities
    "studying", "working", "doing", "trying", "using", "asking", "building", "looking",
    "going", "learning", "taking", "interested", "thinking",
}

_WORD = r"[^\W\d_]+"                       # letters only (also handles accents, e.g. "José")
_NAME_CAPTURE = rf"({_WORD}(?:\s+{_WORD})?)"  # one or two words


def _clean_name(raw: str):
    words = []
    for w in raw.split():
        if w.lower() in _NAME_STOP_WORDS:
            break
        words.append(w)
        if len(words) == 2:
            break
    if not words or any(len(w) < 2 for w in words):
        return None
    return " ".join(w.capitalize() for w in words)


def _extract_name(text: str):
    # Strong patterns: explicit naming
    for pattern in (
        rf"\bmy name(?:\s+is|['’]s)\s+{_NAME_CAPTURE}",
        rf"\bcall me\s+{_NAME_CAPTURE}",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            name = _clean_name(match.group(1))
            if name:
                return name

    # Weak pattern: "I am Sam" / "I'm Sam". Only accepted when the name is
    # capitalized, so "I'm tired" or "I am a student" are never mistaken for names.
    match = re.search(rf"\b[Ii](?:\s+am|['’]m)\s+{_NAME_CAPTURE}", text)
    if match and match.group(1)[0].isupper():
        return _clean_name(match.group(1))

    return None


# --- answer style: only saved when it is clearly a lasting preference -----------
# "Give me a detailed explanation of X" is a one-off request and must NOT be saved.
# "From now on keep answers short" / "I prefer detailed answers" is a preference.
_STYLE_ALWAYS_CUE = re.compile(r"\b(?:from now on|going forward|in the future|in future|always)\b")
_STYLE_SOFT_CUE = re.compile(r"\b(?:prefer|like|want|need|keep|make|give|use)\b")
_STYLE_PLURAL_ANSWERS = re.compile(r"\b(?:answers|responses|replies|explanations)\b")
_STYLE_ADJACENT = re.compile(  # "short answers", "detailed responses"
    r"\b(?:concise|short|shorter|brief|detailed|long|longer|thorough|comprehensive)\s+(?:answers|responses|replies)\b"
)
_STYLE_CONCISE = re.compile(r"\b(?:concise|short|shorter|brief|briefer|to the point)\b")
_STYLE_DETAILED = re.compile(r"\b(?:detailed|in-depth|in depth|thorough|comprehensive|elaborate|long|longer)\b")


def _extract_answer_style(lower_text: str):
    lasting = (
        bool(_STYLE_ALWAYS_CUE.search(lower_text))
        or bool(_STYLE_ADJACENT.search(lower_text))
        or (bool(_STYLE_SOFT_CUE.search(lower_text)) and bool(_STYLE_PLURAL_ANSWERS.search(lower_text)))
    )
    if not lasting:
        return None

    concise = _STYLE_CONCISE.search(lower_text)
    detailed = _STYLE_DETAILED.search(lower_text)
    if concise and detailed:                      # e.g. "short answers, not long ones"
        return "concise" if concise.start() < detailed.start() else "detailed"
    if concise:
        return "concise"
    if detailed:
        return "detailed"
    return None


_LANGUAGE_PATTERN = re.compile(
    r"\b(?:answer|respond|reply|speak|talk)\s+(?:to me\s+)?in\s+(english|bangla|bengali)\b"
)


def extract_profile_updates_from_text(text: str):
    updates = {}
    lower_text = text.lower().strip()

    name = _extract_name(text)
    if name:
        updates["name"] = name

    match = _LANGUAGE_PATTERN.search(lower_text)
    if match:
        updates["preferred_language"] = "English" if match.group(1) == "english" else "Bangla"

    style = _extract_answer_style(lower_text)
    if style:
        updates["preferred_answer_style"] = style

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