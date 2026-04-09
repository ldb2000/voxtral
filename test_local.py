"""Test Voxtral Mini (3B) locally — speech-to-text with latency metrics."""

import argparse
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"


def load_model(device: str = "auto"):
    """Load Voxtral Mini model and processor. Returns load time."""
    print(f"Loading model {MODEL_ID}...")
    start = time.time()

    if device == "mps" and torch.backends.mps.is_available():
        dtype = torch.float32  # MPS doesn't support float16 well for all ops
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
    )

    load_time = time.time() - start
    device_used = next(model.parameters()).device
    print(f"Model loaded on {device_used} ({dtype}) in {load_time:.1f}s")
    return model, processor, load_time


def transcribe(model, processor, audio_path: str) -> dict:
    """Transcribe a single audio file with detailed latency breakdown."""
    import librosa

    # 1. Load audio
    t0 = time.time()
    audio, sr = librosa.load(audio_path, sr=16000)
    t_load = time.time() - t0

    audio_duration = len(audio) / sr

    # 2. Preprocess
    t1 = time.time()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(model.device)
    t_preprocess = time.time() - t1

    # 3. Inference
    t2 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    t_inference = time.time() - t2

    # 4. Decode
    t3 = time.time()
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    t_decode = time.time() - t3

    total = t_load + t_preprocess + t_inference + t_decode

    return {
        "transcription": transcription,
        "audio_duration_s": round(audio_duration, 2),
        "latency": {
            "audio_load_ms": round(t_load * 1000, 1),
            "preprocess_ms": round(t_preprocess * 1000, 1),
            "inference_ms": round(t_inference * 1000, 1),
            "decode_ms": round(t_decode * 1000, 1),
            "total_ms": round(total * 1000, 1),
        },
        "rtf": round(t_inference / audio_duration, 3) if audio_duration > 0 else None,
    }


def print_result(result: dict):
    lat = result["latency"]
    print(f"\n{'='*50}")
    print(f"Audio duration:  {result['audio_duration_s']}s")
    print(f"{'='*50}")
    print(f"  Audio load:    {lat['audio_load_ms']:>8.1f} ms")
    print(f"  Preprocess:    {lat['preprocess_ms']:>8.1f} ms")
    print(f"  Inference:     {lat['inference_ms']:>8.1f} ms")
    print(f"  Decode:        {lat['decode_ms']:>8.1f} ms")
    print(f"  ─────────────────────────────")
    print(f"  Total:         {lat['total_ms']:>8.1f} ms")
    print(f"  RTF:           {result['rtf']:.3f}x (< 1.0 = faster than real-time)")
    print(f"{'='*50}")
    print(f"Transcription:\n{result['transcription']}")


def main():
    parser = argparse.ArgumentParser(description="Test Voxtral Mini STT locally")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument(
        "--warmup", action="store_true",
        help="Run a warmup inference before measuring (more accurate latency)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Repeat transcription N times and show average latency",
    )
    args = parser.parse_args()

    model, processor, _ = load_model(args.device)

    if args.warmup:
        import numpy as np
        print("\nWarmup run...")
        dummy = np.zeros(16000, dtype=np.float32)  # 1s silence
        dummy_inputs = processor(dummy, sampling_rate=16000, return_tensors="pt")
        dummy_inputs = dummy_inputs.to(model.device)
        with torch.no_grad():
            model.generate(**dummy_inputs, max_new_tokens=10)
        print("Warmup done.")

    results = []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\nRun {i+1}/{args.repeat}")
        result = transcribe(model, processor, args.audio)
        results.append(result)
        print_result(result)

    if args.repeat > 1:
        avg_inference = sum(r["latency"]["inference_ms"] for r in results) / len(results)
        avg_total = sum(r["latency"]["total_ms"] for r in results) / len(results)
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        print(f"\n{'='*50}")
        print(f"AVERAGE over {args.repeat} runs:")
        print(f"  Inference:  {avg_inference:.1f} ms")
        print(f"  Total:      {avg_total:.1f} ms")
        print(f"  RTF:        {avg_rtf:.3f}x")


if __name__ == "__main__":
    main()
