import base64
import requests
import os
import time

API_URL = "http://127.0.0.1:8000/v1/detect"

FILES_TO_TEST = [
    '''"/home/deveincosmos/Desktop/VOICE/AI/script1.mp3",
    "/home/deveincosmos/Desktop/VOICE/AI/script2.mp3",
    "/home/deveincosmos/Desktop/VOICE/AI/script3.mp3",
    "/home/deveincosmos/Desktop/VOICE/AI/script4.mp3",
    "/home/deveincosmos/Desktop/VOICE/AI/script5.mp3",
    "/home/deveincosmos/Desktop/VOICE/HUMAN/mallu.mp3",
    "/home/deveincosmos/Desktop/VOICE/HUMAN/nigga.mp3",
    "/home/deveincosmos/Desktop/VOICE/HUMAN/nihita.mp3",
    "/home/deveincosmos/Desktop/VOICE/HUMAN/pranesh.mp3",
    "/home/deveincosmos/Desktop/VOICE/HUMAN/sarvesh.mp3",
    "/home/deveincosmos/Desktop/VOICE/HUMAN/subi.mp3",'''
    "/home/deveincosmos/Desktop/VOICE/AI/script1.mp3",
    "/home/deveincosmos/Desktop/voices/temp.wav"

]

def run_test():
    print("\n" + "=" * 70)
    print(" SIRENVOICE FORENSIC TEST (FINAL)")
    print("=" * 70)

    # Give uvicorn time to bind port
    time.sleep(1.0)

    for path in FILES_TO_TEST:
        if not os.path.exists(path):
            print(f"\n[SKIPPED] Missing file: {path}")
            continue

        with open(path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        try:
            response = requests.post(
                API_URL,
                json={
                    "language": "English",      # you can change later
                    "audioFormat": "mp3",
                    "audioBase64": audio_b64
                },
           timeout=30
            )

        except Exception as e:
            print(f"\n[FILE]: {os.path.basename(path)}")
            print(f"--- REQUEST FAILED: {e}")
            continue

        if response.status_code != 200:
            print(f"\n[FILE]: {os.path.basename(path)}")
            print(f"--- HTTP ERROR: {response.status_code}")
            print(response.text)
            continue

        res = response.json()

        print(f"\n[FILE]: {os.path.basename(path)}")
        print(f"--- Verdict           : {res.get('verdict')}")
        print(f"--- Forensic Score    : {res.get('forensic_score')}")
        print(f"--- Explanation       :")
        print(f"    {res.get('explanation')}")
        print(f"--- Neural Risk       : {res.get('neural')}")
        print(f"--- PCI               : {res.get('pci')}")
        print(f"--- Glottal Asymmetry : {res.get('glottal_asymmetry')}")
        print(f"--- Physical Jerk     : {res.get('physical_jerk')}")
        print(f"--- Spectral Entropy  : {res.get('spectral_entropy')}")

    print("\nTEST SEQUENCE COMPLETE\n")


if __name__ == "__main__":
    run_test()
