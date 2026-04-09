"""Real-time speech-to-text with Voxtral Mini or Whisper."""

import argparse
import io
import sys
import tempfile
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000


def energy(audio: np.ndarray) -> float:
    """RMS energy of audio signal."""
    return float(np.sqrt(np.mean(audio**2)))


class RealtimeSTT:
    def __init__(self, engine: str, language: str, device: str, whisper_model: str):
        self.engine = engine
        self.language = language
        self.device = device
        self.whisper_model = whisper_model
        self.model = None
        self.processor = None

    def load(self):
        if self.engine == "voxtral":
            self._load_voxtral()
        else:
            self._load_whisper()

    def _load_voxtral(self):
        import torch
        from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

        model_id = "mistralai/Voxtral-Mini-3B-2507"
        print(f"Loading {model_id}...")
        dtype = torch.float32
        self.processor = VoxtralProcessor.from_pretrained(model_id)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map=self.device,
        )
        self._model_id = model_id

    def _load_whisper(self):
        import whisper

        print(f"Loading Whisper {self.whisper_model}...")
        self.model = whisper.load_model(self.whisper_model, device=self.device)

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe audio chunk. Returns (text, inference_ms)."""
        if self.engine == "voxtral":
            return self._transcribe_voxtral(audio)
        else:
            return self._transcribe_whisper(audio)

    def _transcribe_voxtral(self, audio: np.ndarray) -> tuple[str, float]:
        import torch

        # Write to temp file (processor needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, SAMPLE_RATE)
            tmp_path = f.name

        t0 = time.time()
        inputs = self.processor.apply_transcription_request(
            language=self.language,
            audio=tmp_path,
            model_id=self._model_id,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elapsed = (time.time() - t0) * 1000

        import os
        os.unlink(tmp_path)
        return text.strip(), elapsed

    def _transcribe_whisper(self, audio: np.ndarray) -> tuple[str, float]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, SAMPLE_RATE)
            tmp_path = f.name

        t0 = time.time()
        result = self.model.transcribe(
            tmp_path, language=self.language,
            without_timestamps=True, fp16=False,
        )
        elapsed = (time.time() - t0) * 1000

        import os
        os.unlink(tmp_path)
        return result["text"].strip(), elapsed


def run_realtime(
    stt: RealtimeSTT,
    silence_threshold: float = 0.01,
    min_speech_s: float = 0.5,
    max_speech_s: float = 10.0,
    silence_after_s: float = 0.8,
):
    """
    Listen to microphone, detect speech segments via energy-based VAD,
    and transcribe each segment.
    """
    chunk_duration = 0.1  # 100ms chunks
    chunk_samples = int(SAMPLE_RATE * chunk_duration)

    speech_buffer = []
    is_speaking = False
    silence_counter = 0
    silence_chunks = int(silence_after_s / chunk_duration)
    min_speech_chunks = int(min_speech_s / chunk_duration)
    max_speech_chunks = int(max_speech_s / chunk_duration)

    segment_count = 0
    transcribing = False

    print(f"\nListening... (silence threshold: {silence_threshold})")
    print(f"  Min speech: {min_speech_s}s | Max segment: {max_speech_s}s | Silence gap: {silence_after_s}s")
    print(f"  Press Ctrl+C to stop.\n")

    def process_segment(audio_data, seg_num):
        nonlocal transcribing
        transcribing = True
        audio = np.concatenate(audio_data)
        duration = len(audio) / SAMPLE_RATE
        text, latency = stt.transcribe(audio)
        if text:
            # Clear line and print result
            sys.stdout.write(f"\r\033[K")
            sys.stdout.write(f"  [{seg_num}] ({duration:.1f}s → {latency:.0f}ms) {text}\n")
            sys.stdout.write(f"\r\033[K  🎤 Listening...")
            sys.stdout.flush()
        transcribing = False

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_speaking, silence_counter, speech_buffer, segment_count

        chunk = indata[:, 0].copy()
        e = energy(chunk)

        if e > silence_threshold:
            if not is_speaking:
                is_speaking = True
                sys.stdout.write(f"\r\033[K  ... speaking ...")
                sys.stdout.flush()
            speech_buffer.append(chunk)
            silence_counter = 0
        elif is_speaking:
            speech_buffer.append(chunk)
            silence_counter += 1

            if silence_counter >= silence_chunks or len(speech_buffer) >= max_speech_chunks:
                if len(speech_buffer) >= min_speech_chunks and not transcribing:
                    segment_count += 1
                    seg_data = list(speech_buffer)
                    seg_num = segment_count
                    # Transcribe in a thread to not block audio
                    t = threading.Thread(
                        target=process_segment, args=(seg_data, seg_num)
                    )
                    t.start()
                speech_buffer = []
                is_speaking = False
                silence_counter = 0

    sys.stdout.write("  🎤 Listening...")
    sys.stdout.flush()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        callback=audio_callback,
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Real-time STT")
    parser.add_argument(
        "--engine", choices=["voxtral", "whisper"], default="whisper",
        help="STT engine (default: whisper — faster for real-time)",
    )
    parser.add_argument("--language", default="fr", help="Language code")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, mps")
    parser.add_argument(
        "--whisper-model", default="base",
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Silence energy threshold (default: 0.01, lower = more sensitive)",
    )
    parser.add_argument(
        "--silence", type=float, default=0.8,
        help="Seconds of silence before segment ends (default: 0.8)",
    )
    parser.add_argument(
        "--max-segment", type=float, default=10.0,
        help="Max segment duration in seconds (default: 10)",
    )
    args = parser.parse_args()

    stt = RealtimeSTT(args.engine, args.language, args.device, args.whisper_model)
    stt.load()
    run_realtime(
        stt,
        silence_threshold=args.threshold,
        silence_after_s=args.silence,
        max_speech_s=args.max_segment,
    )


if __name__ == "__main__":
    main()
