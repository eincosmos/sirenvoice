import base64
import requests
import os

API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "CHANGE_THIS_TO_SECRET"   # must match your FastAPI key

AUDIO_FILE = "/home/deveincosmos/Desktop/VOICE/AI/script1.mp3"

def main():
    if not os.path.exists(AUDIO_FILE):
        print("❌ Audio file not found")
        return

    with open(AUDIO_FILE, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=20
        )
    except Exception as e:
        print("❌ Request failed:", e)
        return

    print("HTTP STATUS:", response.status_code)
    print("RESPONSE JSON:")
    print(response.json())

if __name__ == "__main__":
    main()
