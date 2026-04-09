"""Test Voxtral Mini (3B) via Mistral API."""

import argparse
import base64
import os
import time
from pathlib import Path

from mistralai import Mistral


def transcribe_audio(client: Mistral, audio_path: str) -> dict:
    """Transcribe an audio file using Voxtral Mini via Mistral API."""
    audio_bytes = Path(audio_path).read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Determine MIME type from extension
    ext = Path(audio_path).suffix.lower()
    mime_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    mime_type = mime_types.get(ext, "audio/wav")

    start = time.time()
    response = client.chat.complete(
        model="mistral-audio-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": f"data:{mime_type};base64,{audio_b64}",
                    },
                    {
                        "type": "text",
                        "text": "Transcribe this audio exactly.",
                    },
                ],
            }
        ],
    )
    elapsed = time.time() - start

    return {
        "transcription": response.choices[0].message.content,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
        "elapsed_seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Test Voxtral Mini via Mistral API")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        return

    client = Mistral(api_key=api_key)

    print(f"Transcribing: {args.audio}")
    result = transcribe_audio(client, args.audio)

    print(f"\nModel: {result['model']}")
    print(f"Time: {result['elapsed_seconds']}s")
    print(f"Tokens: {result['usage']}")
    print(f"\nTranscription:\n{result['transcription']}")


if __name__ == "__main__":
    main()
