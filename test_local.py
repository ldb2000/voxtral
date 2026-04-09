"""Test Voxtral Mini (3B) locally with Hugging Face Transformers."""

import argparse
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"


def load_model(device: str = "auto"):
    """Load Voxtral Mini model and processor."""
    print(f"Loading model {MODEL_ID}...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
    )
    return model, processor


def transcribe(model, processor, audio_path: str) -> dict:
    """Transcribe a single audio file."""
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(model.device)

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    elapsed = time.time() - start

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {
        "transcription": transcription,
        "elapsed_seconds": round(elapsed, 2),
        "audio_duration": round(len(audio) / sr, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Test Voxtral Mini locally")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--device", default="auto", help="Device: auto, cpu, cuda, mps"
    )
    args = parser.parse_args()

    model, processor = load_model(args.device)

    print(f"\nTranscribing: {args.audio}")
    result = transcribe(model, processor, args.audio)

    rtf = result["elapsed_seconds"] / result["audio_duration"]
    print(f"Audio duration: {result['audio_duration']}s")
    print(f"Inference time: {result['elapsed_seconds']}s")
    print(f"Real-time factor: {rtf:.2f}x")
    print(f"\nTranscription:\n{result['transcription']}")


if __name__ == "__main__":
    main()
