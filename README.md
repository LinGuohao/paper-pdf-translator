# paper-pdf-translator

`paper-pdf-translator` is a minimal, headless CLI for translating local academic PDFs while preserving the original paper layout as much as possible.

It does not implement PDF layout reconstruction itself. It uses [`pdf2zh-next`](https://github.com/PDFMathTranslate/PDFMathTranslate-next) and its BabelDOC-based PDF pipeline as the underlying engine, and only keeps the local-file CLI flow:

- input: one local PDF path
- translation: one OpenAI-compatible model
- output: one translated PDF file
- mode: monolingual, no watermark, no UI

## Why This Project Exists

The upstream project is powerful, but it includes UI, server, config, and many integration layers. This repository keeps only the capability needed for a paper translation workflow:

- local paper PDF in
- translated PDF out
- OpenAI-compatible API endpoint
- paper-friendly defaults

## Install

```bash
conda activate paper-pdf-translator
pip install -e .
```

On first run, BabelDOC may download assets required for layout processing.

## Usage

```bash
paper-pdf-translator /path/to/paper.pdf \
  /path/to/paper.zh.pdf \
  --api-key "$OPENAI_API_KEY" \
  --base-url "$OPENAI_BASE_URL" \
  --model "$OPENAI_MODEL" \
  --target-lang zh
```

Or equivalently:

```bash
paper-pdf-translator /path/to/paper.pdf \
  --output /path/to/paper.zh.pdf \
  --api-key "$OPENAI_API_KEY"
```

If `--output` is omitted, the tool writes to:

```text
<input-stem>.<target-lang>.translated.pdf
```

in the same directory as the source PDF.

## Default OpenAI-Compatible Backend

The wrapper currently defaults to:

- `OPENAI_BASE_URL=http://10.100.36.32:8000/v1`
- `OPENAI_MODEL=qwen3.5-35b-a3b`

This backend does not require an API key. If no key is provided, the wrapper uses a harmless placeholder value internally for OpenAI-compatible client compatibility.

This wrapper also patches the upstream runtime in two ways for OpenAI-compatible backends:

- skip the eager startup health-check request
- retry transient backend failures such as `502` and other `5xx` responses with exponential backoff

## Environment Variables

These values are picked up automatically if the corresponding CLI flag is not provided:

- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

If `OPENAI_BASE_URL` is not set, the wrapper defaults to `http://10.100.36.32:8000/v1`.
If `OPENAI_MODEL` is not set, the wrapper defaults to `qwen3.5-35b-a3b`.

## Useful Options

```bash
paper-pdf-translator paper.pdf \
  --target-lang zh \
  --source-lang en \
  --pages 1-5 \
  --qps 4 \
  --pool-max-workers 20 \
  --primary-font-family serif \
  --auto-ocr \
  --compatibility-mode
```

Notes:

- `--auto-ocr` is useful for scanned or OCR-poor papers, but may reduce fidelity on normal digital PDFs.
- `--compatibility-mode` enables upstream compatibility-oriented options. Use it only when a PDF renders badly with the default mode.
- Automatic glossary extraction is disabled by default in this wrapper to reduce extra LLM calls and keep the flow simple.

## Scope

This repository intentionally does not include:

- Web UI
- Gradio app
- HTTP server
- Zotero integration
- batch folder workflows

## License

This repository contains only the wrapper code in this repo. The underlying `pdf2zh-next` dependency keeps its own upstream license and terms.
