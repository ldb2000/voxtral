"""Benchmark Voxtral Mini on multiple audio files."""

import argparse
import json
from pathlib import Path

from test_local import load_model, transcribe


def main():
    parser = argparse.ArgumentParser(description="Benchmark Voxtral Mini")
    parser.add_argument(
        "--audio-dir", required=True, help="Directory containing audio files"
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--output", default="results.json", help="Output JSON file")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    extensions = {".wav", ".mp3", ".flac", ".ogg"}
    audio_files = sorted(
        f for f in audio_dir.iterdir() if f.suffix.lower() in extensions
    )

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    model, processor = load_model(args.device)

    results = []
    total_audio = 0.0
    total_inference = 0.0

    for audio_path in audio_files:
        print(f"\nProcessing: {audio_path.name}")
        result = transcribe(model, processor, str(audio_path))
        result["file"] = audio_path.name
        results.append(result)

        total_audio += result["audio_duration"]
        total_inference += result["elapsed_seconds"]

        rtf = result["elapsed_seconds"] / result["audio_duration"]
        print(f"  Duration: {result['audio_duration']}s | Inference: {result['elapsed_seconds']}s | RTF: {rtf:.2f}x")

    avg_rtf = total_inference / total_audio if total_audio > 0 else 0
    summary = {
        "total_files": len(results),
        "total_audio_seconds": round(total_audio, 2),
        "total_inference_seconds": round(total_inference, 2),
        "average_rtf": round(avg_rtf, 3),
        "results": results,
    }

    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary: {len(results)} files | {total_audio:.1f}s audio | avg RTF: {avg_rtf:.2f}x")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
