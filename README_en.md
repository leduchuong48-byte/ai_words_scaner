# AI Vocabulary Scanner (words_scaner)

[中文](README.md)

A `Gradio + spaCy + LLM` based tool for extracting vocabulary from English reading materials.

## Features

- Extract candidate words from `PDF / EPUB / TXT`
- Two filtering modes: professional reading (blacklist) and IELTS prep (whitelist)
- Use OpenAI-compatible / Ollama / Gemini models for contextual meanings and collocations
- Export to `Excel / PDF / TXT`

## Requirements

- Python `3.11+`
- Linux / macOS / WSL recommended
- Optional: Docker + Docker Compose

## Quick Start

```bash
pip install -r requirements.txt
python3 app.py
```

Or with Docker:

```bash
docker compose up --build
```

## Security & Privacy

- Never commit real API keys
- Never commit private input files or generated outputs with sensitive data
- Keep local secrets in ignored files such as `.env`

## Disclaimer

By using this project, you acknowledge and agree to the [Disclaimer](DISCLAIMER.md).

## License

See [LICENSE](LICENSE).
