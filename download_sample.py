"""Download a sample audio file for testing."""

import urllib.request
from pathlib import Path

# LibriSpeech sample — public domain English speech
SAMPLE_URL = "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"
OUTPUT_PATH = Path("sample.wav")


def main():
    if OUTPUT_PATH.exists():
        print(f"{OUTPUT_PATH} already exists, skipping download.")
        return

    print(f"Downloading sample audio to {OUTPUT_PATH}...")
    urllib.request.urlretrieve(SAMPLE_URL, OUTPUT_PATH)
    print(f"Done. Use it with:\n  python test_local.py --audio {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
