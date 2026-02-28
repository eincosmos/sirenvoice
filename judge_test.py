import base64
import requests
import os

# 🔹 LIVE API ENDPOINT (must include the route)
API_URL = "https://sirenvoice.onrender.com/api/voice-detection"

# 🔹 MUST MATCH Render → Environment Variable
API_KEY = "sirenvoice_9f3K2mA7xQpL_2026"

# 🔹 Path to your test MP3 file
AUDIO_FILE = "/home/deveincosmos/Desktop/VOICE/AI/script1.mp3"


def main():
    if not os.path.exists(AUDIO_FILE):
        print("❌ Audio file not found:", AUDIO_FILE)
        return

    # Read audio
    with open(AUDIO_FILE, "rb") as f:
        audio_bytes = f.read()

    # Encode to base64
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Payload must match FastAPI request model
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }

    # Headers (VERY IMPORTANT)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=360 # allow cold start + lazy model load
        )
    except Exception as e:
        print("❌ Request failed:", e)
        return

    print("\nHTTP STATUS:", response.status_code)
    print("RESPONSE JSON:")

    try:
        print(response.json())
    except Exception:
        print(response.text)


if __name__ == "__main__":
    main()
