from __future__ import annotations

import base64
import json
import os
import threading

from flask import Flask, jsonify, render_template, request
from flask_sock import Sock
from groq import Groq

from commands import process, reload_kb

app = Flask(__name__)
sock = Sock(app)

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ---------- Groq client (shared) ----------
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---------- Configuration ----------
# Use a currently supported Groq model.
# gpt-oss-120b is the recommended replacement for the retired llama-3.3-70b-versatile.
GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
#  Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/command", methods=["POST"])
def command():
    """Text path: knowledge base first, then Groq for anything else."""
    data = request.get_json(silent=True) or {}
    raw = (data.get("command") or "").strip()

    if not raw:
        return jsonify({"text": "I didn't catch that.", "open_url": None, "exit": False})

    if len(raw) > 500:
        return jsonify({"text": "That command is too long.", "open_url": None, "exit": False})

    # 1. Local knowledge base first (fast, free)
    resp = process(raw)

    # 2. If fallback, ask Groq
    if resp.intent_id == "fallback":
        answer = ask_groq(raw)
        if answer:
            resp.text = answer
            resp.intent_id = "groq"
            resp.confidence = 0.7

    return jsonify({
        "text": resp.text,
        "open_url": resp.open_url,
        "exit": resp.exit,
        "intent": resp.intent_id,
        "confidence": resp.confidence,
    })


@app.route("/api/reload", methods=["POST"])
def reload():
    count = reload_kb()
    return jsonify({"ok": True, "intents": count})


# ============================================================
#  WebSocket bridge: Browser <-> Deepgram Voice Agent
# ============================================================
@sock.route("/ws/voice")
def voice_websocket(ws):
    """
    Bridges the browser's mic audio to Deepgram's Voice Agent API,
    which orchestrates STT + LLM (Groq) + TTS.
    """
    if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
        ws.send(json.dumps({"type": "error", "message": "Missing DEEPGRAM_API_KEY or GROQ_API_KEY"}))
        return

    # ---- import Deepgram SDK pieces lazily ----
    try:
        from deepgram import DeepgramClient
        from deepgram.core.events import EventType
        from deepgram.agent.v1.types import (
            AgentV1Settings,
            AgentV1SettingsAgent,
            AgentV1SettingsAudio,
            AgentV1SettingsAudioInput,
            AgentV1SettingsAudioOutput,
            AgentV1SettingsAgentListen,
            AgentV1SettingsAgentListenProvider_V1,
        )
        from deepgram.types.think_settings_v1 import ThinkSettingsV1
        from deepgram.types.think_settings_v1provider import ThinkSettingsV1Provider_OpenAi
        from deepgram.types.speak_settings_v1 import SpeakSettingsV1
        from deepgram.types.speak_settings_v1provider import SpeakSettingsV1Provider_Deepgram
    except ImportError as e:
        ws.send(json.dumps({"type": "error", "message": f"Deepgram SDK missing pieces: {e}"}))
        return

    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    # ---- forward Deepgram events to the browser ----
    def on_message(message):
        try:
            # Binary audio frames come through as bytes
            if isinstance(message, (bytes, bytearray)):
                ws.send(json.dumps({
                    "type": "audio",
                    "data": base64.b64encode(bytes(message)).decode("ascii"),
                }))
                return

            msg_type = getattr(message, "type", "unknown")
            payload = {"type": "deepgram_event", "event": msg_type}

            # Transcript from the user (STT result)
            if hasattr(message, "transcript") and message.transcript:
                payload["transcript"] = message.transcript
                payload["role"] = "user"
                ws.send(json.dumps(payload))
                return

            # Assistant's text reply
            if hasattr(message, "content") and message.content:
                payload["content"] = message.content
                payload["role"] = "assistant"
                ws.send(json.dumps(payload))
                return

            # Other event types — just log them
            ws.send(json.dumps(payload))
        except Exception as e:
            print(f"[bridge] forward error: {e}")

    try:
        with client.agent.v1.connect() as connection:
            settings = AgentV1Settings(
                audio=AgentV1SettingsAudio(
                    input=AgentV1SettingsAudioInput(
                        encoding="linear16",
                        sample_rate=16000,
                    ),
                    output=AgentV1SettingsAudioOutput(
                        encoding="linear16",
                        sample_rate=24000,
                    ),
                ),
                agent=AgentV1SettingsAgent(
                    listen=AgentV1SettingsAgentListen(
                        provider=AgentV1SettingsAgentListenProvider_V1(
                            type="deepgram",
                            model="nova-3",
                        )
                    ),
                    think=ThinkSettingsV1(
                        provider=ThinkSettingsV1Provider_OpenAi(
                            type="open_ai",
                            model=GROQ_MODEL,
                            temperature=0.7,
                            endpoint="https://api.groq.com/openai/v1/chat/completions",
                            api_key=GROQ_API_KEY,
                        ),
                        prompt=(
                            "You are a helpful voice assistant. "
                            "Keep answers concise and conversational — under 2 sentences. "
                            "Do not use markdown, lists, or emoji."
                        ),
                    ),
                    speak=SpeakSettingsV1(
                        provider=SpeakSettingsV1Provider_Deepgram(
                            type="deepgram",
                            model="aura-2-thalia-en",
                        )
                    ),
                    greeting="Hello! How can I help you today?",
                ),
            )

            connection.send_settings(settings)
            connection.on(EventType.MESSAGE, on_message)

            # Run the listener on its own thread
            listener = threading.Thread(target=connection.start_listener, daemon=True)
            listener.start()

            ws.send(json.dumps({"type": "connected"}))

            # Forward browser audio chunks -> Deepgram
            while True:
                data = ws.receive()
                if data is None:
                    break

                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if parsed.get("type") == "audio":
                    chunk = base64.b64decode(parsed["data"])
                    connection.send_audio(chunk)
                elif parsed.get("type") == "stop":
                    break

    except Exception as e:
        print(f"[voice] error: {e}")
        try:
            ws.send(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


# ============================================================
#  Helpers
# ============================================================
def ask_groq(question: str) -> str | None:
    """Ask Groq for a concise, assistant-style answer."""
    if not groq_client:
        print("[groq] no client — GROQ_API_KEY not set")
        return None
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful voice assistant. "
                        "Keep answers short and conversational — under 2 sentences. "
                        "Do not use markdown, lists, or emoji."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[groq] error: {e}")
        return None


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    # Warn early if keys are missing
    if not DEEPGRAM_API_KEY:
        print("[startup] WARNING: DEEPGRAM_API_KEY is not set (voice will not work).")
    if not GROQ_API_KEY:
        print("[startup] WARNING: GROQ_API_KEY is not set (LLM answers will not work).")
    else:
        print(f"[startup] Groq client ready (model: {GROQ_MODEL}).")

    app.run(host="127.0.0.1", port=5000, debug=True)