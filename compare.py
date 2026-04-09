"""Compare Voxtral Mini vs Whisper — same audio, side-by-side latency."""

import argparse
import time

import torch


def run_voxtral(audio_path: str, language: str, device: str) -> dict:
    """Run Voxtral Mini STT."""
    from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

    MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"

    # Load
    t0 = time.time()
    if device == "mps":
        dtype = torch.float32
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    processor = VoxtralProcessor.from_pretrained(MODEL_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device,
    )
    t_load = time.time() - t0

    # Preprocess
    t1 = time.time()
    inputs = processor.apply_transcription_request(
        language=language,
        audio=audio_path,
        model_id=MODEL_ID,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    t_preprocess = time.time() - t1

    # Inference
    t2 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    t_inference = time.time() - t2

    # Decode
    t3 = time.time()
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    t_decode = time.time() - t3

    return {
        "model": f"Voxtral Mini 3B",
        "transcription": transcription,
        "model_load_ms": round(t_load * 1000, 1),
        "preprocess_ms": round(t_preprocess * 1000, 1),
        "inference_ms": round(t_inference * 1000, 1),
        "decode_ms": round(t_decode * 1000, 1),
        "total_ms": round((t_preprocess + t_inference + t_decode) * 1000, 1),
    }


def run_whisper(audio_path: str, language: str, device: str, model_size: str) -> dict:
    """Run OpenAI Whisper STT."""
    import whisper

    # Load
    t0 = time.time()
    model = whisper.load_model(model_size, device=device)
    t_load = time.time() - t0

    # Transcribe (handles mel channels automatically for all model sizes)
    t1 = time.time()
    result = model.transcribe(
        audio_path, language=language, without_timestamps=True, fp16=False,
    )
    t_total = time.time() - t1

    return {
        "model": f"Whisper {model_size}",
        "transcription": result["text"],
        "model_load_ms": round(t_load * 1000, 1),
        "preprocess_ms": 0,
        "inference_ms": round(t_total * 1000, 1),
        "decode_ms": 0,
        "total_ms": round(t_total * 1000, 1),
    }


def print_comparison(results: list[dict], audio_duration: float):
    w = 60
    print(f"\n{'='*w}")
    print(f"  COMPARISON — {audio_duration:.1f}s audio")
    print(f"{'='*w}")

    # Header
    names = [r["model"] for r in results]
    print(f"{'':>20s}", end="")
    for name in names:
        print(f"  {name:>16s}", end="")
    print()
    print(f"  {'─'*(w-4)}")

    # Metrics
    for metric, key in [
        ("Model load", "model_load_ms"),
        ("Preprocess", "preprocess_ms"),
        ("Inference", "inference_ms"),
        ("Decode", "decode_ms"),
        ("Total (no load)", "total_ms"),
    ]:
        print(f"  {metric:>18s}", end="")
        for r in results:
            print(f"  {r[key]:>13.1f} ms", end="")
        print()

    # RTF
    print(f"  {'RTF':>18s}", end="")
    for r in results:
        rtf = r["inference_ms"] / (audio_duration * 1000)
        print(f"  {rtf:>14.3f}x", end="")
    print()

    # Speedup
    if len(results) == 2:
        r0, r1 = results
        speedup = r0["inference_ms"] / r1["inference_ms"] if r1["inference_ms"] > 0 else float("inf")
        faster = r1["model"] if speedup > 1 else r0["model"]
        ratio = max(speedup, 1/speedup) if speedup > 0 else float("inf")
        print(f"\n  → {faster} is {ratio:.1f}x faster on inference")

    print(f"\n{'='*w}")
    for r in results:
        print(f"\n  [{r['model']}]")
        print(f"  {r['transcription']}")
    print(f"\n{'='*w}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Voxtral Mini vs Whisper on the same audio"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--language", default="fr", help="Language code (en, fr, etc.)")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, mps")
    parser.add_argument(
        "--whisper-model", default="base",
        help="Whisper model size: tiny, base, small, medium, large (default: base)",
    )
    args = parser.parse_args()

    # Get audio duration
    import librosa
    audio, sr = librosa.load(args.audio, sr=16000)
    audio_duration = len(audio) / sr

    print(f"Audio: {args.audio} ({audio_duration:.1f}s)")
    print(f"Language: {args.language}")
    print(f"Device: {args.device}")

    # Run Voxtral
    print(f"\n--- Running Voxtral Mini 3B ---")
    voxtral_result = run_voxtral(args.audio, args.language, args.device)

    # Run Whisper
    print(f"\n--- Running Whisper {args.whisper_model} ---")
    whisper_result = run_whisper(args.audio, args.language, args.device, args.whisper_model)

    # Compare
    print_comparison([voxtral_result, whisper_result], audio_duration)


if __name__ == "__main__":
    main()
