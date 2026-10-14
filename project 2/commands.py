"""
Intent-based command processor — JSON knowledge base.

- Loads intents from data/knowledge.json
- Matches user input with weighted, token-aware scoring
- Routes searches to the right site (YouTube, Google, GitHub, etc.)
- Returns a structured Response the web layer can render and speak
"""

from __future__ import annotations

import datetime
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus


# ============================================================
#  Response shape
# ============================================================

@dataclass
class Response:
    text: str
    open_url: str | None = None
    exit: bool = False
    intent_id: str = ""
    confidence: float = 0.0
    meta: dict = field(default_factory=dict)


# ============================================================
#  Knowledge base
# ============================================================

_KB_PATH = Path(__file__).parent / "data" / "knowledge.json"
_intents: list[dict] = []


def _load_kb() -> list[dict]:
    """Load and validate the knowledge base once at import time."""
    if not _KB_PATH.exists():
        print(f"[kb] WARNING: {_KB_PATH} not found. Only time/date/search will work.")
        return []

    try:
        with _KB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[kb] JSON parse error in {_KB_PATH.name}: {e}")
        print(f"[kb] line {e.lineno}, column {e.colno}: {e.msg}")
        return []

    intents = data.get("intents", [])
    cleaned = []
    for it in intents:
        if not isinstance(it, dict):
            continue
        if "patterns" not in it or "response" not in it:
            print(f"[kb] skipping malformed intent: {it.get('id', '?')}")
            continue
        it.setdefault("kind", "info")
        it.setdefault("priority", 50)
        it.setdefault("id", f"intent_{len(cleaned)}")
        it["patterns"] = [str(p).lower().strip() for p in it["patterns"]]
        cleaned.append(it)

    print(f"[kb] loaded {len(cleaned)} intents from {_KB_PATH.name}")
    return cleaned


def reload_kb() -> int:
    """Reload the knowledge base at runtime. Returns the new intent count."""
    global _intents
    _intents = _load_kb()
    return len(_intents)


_intents = _load_kb()


# ============================================================
#  Matching
# ============================================================

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _score(intent: dict, user_tokens: list[str], raw: str) -> float:
    """Score how well an intent matches the user's input. Higher = better."""
    user_set = set(user_tokens)
    best = 0.0

    for pattern in intent["patterns"]:
        p_tokens = _tokens(pattern)
        if not p_tokens:
            continue

        # 1. Full phrase appears verbatim in the raw input
        if pattern in raw:
            best = max(best, 1.0)
            continue

        # 2. All tokens of the pattern present in the user's input
        if set(p_tokens).issubset(user_set):
            weight = min(1.0, 0.6 + 0.1 * len(p_tokens))
            best = max(best, weight)
            continue

        # 3. Partial token overlap
        overlap = len(set(p_tokens) & user_set) / len(p_tokens)
        if overlap >= 0.66:
            best = max(best, 0.4 + 0.2 * overlap)

    return best


def _pick_response(intent: dict) -> str:
    """Support a list of responses — pick one at random for variety."""
    resp = intent["response"]
    if isinstance(resp, list):
        return random.choice(resp)
    return str(resp)


# ============================================================
#  Dynamic responses
# ============================================================

def _dynamic_time() -> str:
    now = datetime.datetime.now()
    h = now.strftime("%I").lstrip("0") or "12"
    m = now.strftime("%M")
    ap = now.strftime("%p")
    return f"It's currently {h}:{m} {ap}."


def _dynamic_date() -> str:
    today = datetime.datetime.now().strftime("%A, %B %d, %Y")
    return f"Today is {today}."


# ============================================================
#  Search routing
# ============================================================

# Map of site keywords -> (Display Name, URL template with {q} placeholder)
SEARCH_SITES = {
    "youtube":       ("YouTube",        "https://www.youtube.com/results?search_query={q}"),
    "google":        ("Google",         "https://www.google.com/search?q={q}"),
    "github":        ("GitHub",         "https://github.com/search?q={q}"),
    "wikipedia":     ("Wikipedia",      "https://en.wikipedia.org/w/index.php?search={q}"),
    "amazon":        ("Amazon",         "https://www.amazon.com/s?k={q}"),
    "maps":          ("Google Maps",    "https://www.google.com/maps/search/{q}"),
    "spotify":       ("Spotify",        "https://open.spotify.com/search/{q}"),
    "reddit":        ("Reddit",         "https://www.reddit.com/search/?q={q}"),
    "twitter":       ("Twitter",        "https://twitter.com/search?q={q}"),
    "x":             ("X",              "https://twitter.com/search?q={q}"),
    "stackoverflow": ("Stack Overflow", "https://stackoverflow.com/search?q={q}"),
    "flipkart":      ("Flipkart",       "https://www.flipkart.com/search?q={q}"),
    "netflix":       ("Netflix",        "https://www.netflix.com/search?q={q}"),
    "instagram":     ("Instagram",      "https://www.instagram.com/explore/tags/{q}"),
    "pinterest":     ("Pinterest",      "https://www.pinterest.com/search/pins/?q={q}"),
    "imdb":          ("IMDb",           "https://www.imdb.com/find?q={q}"),
    "quora":         ("Quora",          "https://www.quora.com/search?q={q}"),
    "medium":        ("Medium",         "https://medium.com/search?q={q}"),
}


def _handle_search(cmd: str) -> Response | None:
    """
    Route search intents to the right site.

    Examples:
      "play music on youtube of karan aujla"    -> YouTube search for "karan aujla"
      "youtube karan aujla"                     -> YouTube search
      "find python tutorials on youtube"        -> YouTube search
      "google machine learning"                 -> Google search
      "search reddit for best pizza"            -> Reddit search
      "look up nikola tesla on wikipedia"       -> Wikipedia search
      "search for python async"                 -> Google search
    """
    raw = cmd.strip()

    # ---- 1. Which site? (default = Google) ----
    target_name = "Google"
    url_template = SEARCH_SITES["google"][1]

    for key, (name, tmpl) in SEARCH_SITES.items():
        # \b word boundary so "youtube" matches but not "youtuber" falsely
        if re.search(rf"\b{re.escape(key)}\b", raw, flags=re.IGNORECASE):
            target_name = name
            url_template = tmpl
            break

    # ---- 2. Strip filler words to isolate the query ----
    filler_patterns = [
        r"^\s*(?:please\s+)?",
        r"\b(?:can you|could you|would you|i want to|i wanna|i'd like to)\b",
        r"\b(?:play|search|find|look\s*up|google|show me|open|go to|put on|give me)\b",
        r"\b(?:music|song|songs|video|videos|track|tracks|playlist|movie|movies)\b",
        r"\b(?:on|in|from|at|for|about|of)\s+(?:youtube|google|github|wikipedia|spotify|reddit|amazon|maps|twitter|x|flipkart|stackoverflow|netflix|instagram|pinterest|imdb|quora|medium)\b",
        r"\b(?:on|in|from|at|for|about)\b",
        r"\b(?:youtube|google|github|wikipedia|spotify|reddit|amazon|maps|twitter|x|flipkart|stackoverflow|netflix|instagram|pinterest|imdb|quora|medium)\b",
    ]

    query = raw
    for pat in filler_patterns:
        query = re.sub(pat, " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query).strip(" ?.!,")

    # ---- 3. If nothing meaningful remains, ask ----
    if not query or len(query) < 2:
        if re.search(r"\b(search|find|look\s*up|play|google)\b", raw, re.IGNORECASE):
            return Response(
                f"What would you like me to search for on {target_name}?",
                intent_id="search_web",
            )
        return None

    # ---- 4. Build and return ----
    url = url_template.format(q=quote_plus(query))

    return Response(
        f"Searching {target_name} for {query}.",
        open_url=url,
        intent_id="search_web",
        confidence=0.95,
    )


# ============================================================
#  Main entry point
# ============================================================

def process(cmd: str) -> Response:
    """Turn a raw command string into a Response."""
    if not cmd or not cmd.strip():
        return Response("I didn't catch that. Could you repeat it?", confidence=0.0)

    raw = cmd.lower().strip()
    # Strip trailing punctuation that confuses matching
    raw = re.sub(r"[?!.]+$", "", raw).strip()
    tokens = _tokens(raw)

    # ---- 1. Search/play intents get priority — they're actions with payloads ----
    # Only attempt search routing if the input looks like a search command.
    if re.search(
        r"\b(search|find|look\s*up|play|google|show me|put on)\b",
        raw,
        flags=re.IGNORECASE,
    ):
        search_resp = _handle_search(raw)
        if search_resp:
            return search_resp

    # ---- 2. Score all intents from the knowledge base ----
    scored: list[tuple[float, dict]] = []
    for intent in _intents:
        s = _score(intent, tokens, raw)
        if s > 0:
            scored.append((s + intent["priority"] / 1000.0, intent))

    if not scored:
        return _fallback(raw)

    scored.sort(key=lambda x: x[0], reverse=True)
    score, best = scored[0]

    # ---- 3. Dynamic responses ----
    resp_text = _pick_response(best)
    if resp_text == "__DYNAMIC_TIME__":
        resp_text = _dynamic_time()
    elif resp_text == "__DYNAMIC_DATE__":
        resp_text = _dynamic_date()

    # ---- 4. Build Response ----
    r = Response(
        text=resp_text,
        intent_id=best["id"],
        confidence=round(min(score, 1.0), 2),
    )

    kind = best.get("kind", "info")

    if kind == "site" and best.get("url"):
        r.open_url = best["url"]

    if best["id"] == "exit":
        r.exit = True

    return r


def _fallback(raw: str) -> Response:
    """Professional fallback when nothing matches."""
    return Response(
        text=(
            "I'm not sure how to help with that yet. "
            "I can tell you the time or date, open sites like Google and YouTube, "
            "run a web search, or answer questions from my knowledge base. "
            "You can also say 'what can you do' for a full list."
        ),
        intent_id="fallback",
        confidence=0.0,
    )