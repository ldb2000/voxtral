# Voxtral Mini (3B) - Test Repository

Testing Mistral AI's Voxtral Mini, a 3B parameter speech-to-text model.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Via Mistral API

```bash
export MISTRAL_API_KEY="your-api-key"
python test_api.py --audio sample.wav
```

### Local inference with Transformers

```bash
python test_local.py --audio sample.wav
```

## Scripts

- `test_api.py` — Test Voxtral Mini via Mistral's API
- `test_local.py` — Run Voxtral Mini locally with Hugging Face Transformers
- `benchmark.py` — Benchmark transcription on a set of audio files
