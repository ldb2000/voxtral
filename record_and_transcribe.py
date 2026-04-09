"""Record audio from microphone, then transcribe with Voxtral Mini."""

import argparse
import tempfile
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from test_local import load_model, transcribe, print_result


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from default microphone."""
    print(f"\nRecording {duration}s... Speak now!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording done.")
    return audio.flatten()


def record_until_enter(sample_rate: int = 16000) -> np.ndarray:
    """Record audio until user presses Enter."""
    import threading

    print("\nRecording... Press Enter to stop.")
    chunks = []
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        if not stop_event.is_set():
            chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32", callback=callback
    )
    stream.start()
    input()  # Wait for Enter
    stop_event.set()
    stream.stop()
    stream.close()

    if not chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks).flatten()
    print(f"Recorded {len(audio) / sample_rate:.1f}s of audio.")
    return audio


def main():
    parser = argparse.ArgumentParser(
        description="Record from mic & transcribe with Voxtral Mini"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Recording duration in seconds (omit to record until Enter)",
    )
    parser.add_argument("--language", default="en", help="Language code (en, fr, etc.)")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--save", default=None, help="Save recording to this WAV path")
    args = parser.parse_args()

    # Load model first so it's ready when recording finishes
    model, processor, load_time = load_model(args.device)

    # Record
    if args.duration:
        audio = record_audio(args.duration)
    else:
        audio = record_until_enter()

    if len(audio) == 0:
        print("No audio recorded.")
        return

    # Save to temp file (or specified path)
    wav_path = args.save or tempfile.mktemp(suffix=".wav")
    sf.write(wav_path, audio, 16000)

    if args.save:
        print(f"Saved recording to {wav_path}")

    # Transcribe and show latency
    t_start = time.time()
    result = transcribe(model, processor, wav_path, language=args.language)
    e2e = time.time() - t_start

    print_result(result)
    print(f"\nEnd-to-end (record stop → result): {e2e*1000:.0f} ms")

    # Cleanup temp file
    if not args.save:
        import os
        os.unlink(wav_path)


if __name__ == "__main__":
    main()
