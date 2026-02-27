# AI Vocabulary Scanner (ai_words_scaner)

[![Docker Pulls](https://img.shields.io/docker/pulls/leduchuong/ai_words_scaner?logo=docker&style=flat-square)](https://hub.docker.com/r/leduchuong/ai_words_scaner)
[![Docker Stars](https://img.shields.io/docker/stars/leduchuong/ai_words_scaner?logo=docker&style=flat-square)](https://hub.docker.com/r/leduchuong/ai_words_scaner)
[![GitHub Stars](https://img.shields.io/github/stars/leduchuong48-byte/ai_words_scaner?style=flat-square)](https://github.com/leduchuong48-byte/ai_words_scaner/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/leduchuong48-byte/ai_words_scaner?style=flat-square)](https://github.com/leduchuong48-byte/ai_words_scaner/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/leduchuong48-byte/ai_words_scaner?style=flat-square)](https://github.com/leduchuong48-byte/ai_words_scaner/issues)
[![License](https://img.shields.io/github/license/leduchuong48-byte/ai_words_scaner?style=flat-square)](https://github.com/leduchuong48-byte/ai_words_scaner/blob/main/LICENSE)

[中文](README.md)

A practical AI pipeline for English reading vocabulary: upload `PDF/EPUB/TXT`, extract candidate words, enrich them with LLM contextual meanings and collocations, then export review-ready outputs.

## Why this tool

Most vocabulary workflows are fragmented: extraction, filtering, explanation, collocation lookup, and export all happen in different tools. `ai_words_scaner` unifies this into one repeatable WebUI flow.

## Real Value Propositions

- End-to-end processing from one upload (`PDF / EPUB / TXT`).
- Dual filtering strategy:
  - Professional Reading Mode (blacklist) to remove common words.
  - IELTS Prep Mode (whitelist) to focus on exam vocabulary.
- Multi-provider model center: OpenAI official/compatible, OpenAI Responses, Gemini, Ollama local.
- Stability-first execution: checkpoint resume, retries, bounded concurrency, and circuit breaker.
- Useful outputs: `Excel / PDF / TXT wordbook` plus Typing-World CSV.

## For Portainer/Synology Users

Copy this into Portainer stacks and hit Deploy. Done.

## Docker Compose

```yaml
services:
  ai_words_scaner:
    image: leduchuong/ai_words_scaner:latest
    container_name: ai_words_scaner
    restart: unless-stopped
    ports:
      - "1016:7860"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    command: ["python", "app.py"]
```

## Quick Start

### Docker (one command)

```bash
docker run --rm -it \
  -p 1016:7860 \
  --add-host host.docker.internal:host-gateway \
  leduchuong/ai_words_scaner:latest
```

Then open: `http://<your-host-ip>:1016`

### Run from source

```bash
pip install -r requirements.txt
python3 app.py
```

## Typical Workflow

1. Configure provider/model in **Model Center** and set default model.
2. Upload your ebook (`PDF/EPUB/TXT`) in **Task Board**.
3. Choose filter strategy and output formats.
4. Start the task and download exported files.

## Model & Config Capabilities

- Provider CRUD and default-model switching.
- Online model-list fetch for OpenAI-compatible APIs, Gemini, and Ollama.
- `llm_settings.json` is sanitized template-only by default.

## Stability Design

- Checkpoint resume via `cache_results.jsonl`.
- Async concurrency pool for throughput.
- Retry transient failures and circuit-break on continuous fatal errors.

## Output Files

- `vocabulary_reading.xlsx`
- `vocabulary_reading.pdf`
- `maimemo_vocabulary.txt`
- `typing_world.csv`

## Profile Style Metrics

<p align="left"><img src="https://komarev.com/ghpvc/?username=leduchuong48-byte&label=Repo%20views&color=0e75b6&style=flat" alt="views" /></p>

<p>
  <img align="left" src="https://github-readme-stats.vercel.app/api/top-langs?username=leduchuong48-byte&show_icons=true&locale=en&layout=compact" alt="top-langs" />
  <img align="center" src="https://github-readme-stats.vercel.app/api?username=leduchuong48-byte&show_icons=true&locale=en" alt="stats" />
</p>

<p><img align="center" src="https://github-readme-streak-stats.herokuapp.com/?user=leduchuong48-byte" alt="streak" /></p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=leduchuong48-byte/ai_words_scaner&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=leduchuong48-byte/ai_words_scaner&type=Date" />
  <img alt="Star History" src="https://api.star-history.com/svg?repos=leduchuong48-byte/ai_words_scaner&type=Date" />
</picture>

## License

MIT. See [LICENSE](LICENSE).

## Disclaimer

By using this project, you acknowledge and agree to the [Disclaimer](DISCLAIMER.md).
