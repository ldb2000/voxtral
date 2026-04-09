# Voxtral Mini (3B) — Local STT Latency Test

Test de speech-to-text local avec [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) de Mistral AI. Mesure détaillée de la latence (chargement audio, preprocessing, inférence, décodage).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Télécharger un sample audio

```bash
python download_sample.py
```

### 2. Transcrire un fichier audio

```bash
python test_local.py --audio sample.wav
```

Options :
- `--device mps` — forcer le device (auto, cpu, cuda, mps)
- `--warmup` — run à blanc pour des mesures plus précises
- `--repeat 5` — répéter N fois et afficher la latence moyenne

Exemple complet :
```bash
python test_local.py --audio sample.wav --device mps --warmup --repeat 3
```

### 3. Enregistrer depuis le micro et transcrire

```bash
# Enregistrement libre (appuyer sur Entrée pour arrêter)
python record_and_transcribe.py

# Enregistrement de 5 secondes
python record_and_transcribe.py --duration 5

# Sauvegarder l'enregistrement
python record_and_transcribe.py --duration 5 --save recording.wav
```

### 4. Benchmark sur un dossier

```bash
python benchmark.py --audio-dir ./audio_samples/ --device mps
```

## Métriques de latence

Le script affiche :
- **Audio load** — temps de chargement du fichier audio
- **Preprocess** — tokenization / feature extraction
- **Inference** — génération des tokens (le gros du travail)
- **Decode** — décodage tokens → texte
- **RTF** (Real-Time Factor) — ratio inference/durée audio (< 1.0 = plus rapide que le temps réel)

## Scripts

| Script | Description |
|---|---|
| `test_local.py` | Transcription locale avec métriques de latence |
| `record_and_transcribe.py` | Enregistrement micro → transcription |
| `benchmark.py` | Benchmark batch sur un dossier audio |
| `download_sample.py` | Télécharge un sample audio de test |
| `test_api.py` | Test via l'API Mistral (optionnel) |
