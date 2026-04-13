from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from paper_pdf_translator import __version__
from paper_pdf_translator.translator import TranslateRequest
from paper_pdf_translator.translator import translate_pdf


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return float(value)


def _default_output_path(input_pdf: Path, target_lang: str) -> Path:
    return input_pdf.with_name(f"{input_pdf.stem}.{target_lang}.translated.pdf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-pdf-translator",
        description=(
            "Translate a local academic PDF with an OpenAI-compatible model while "
            "preserving the original paper layout as much as possible."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("input_pdf", type=Path, help="Path to the source PDF.")
    parser.add_argument(
        "output_pdf",
        nargs="?",
        type=Path,
        help=(
            "Optional output PDF path as a positional argument. "
            "Equivalent to --output."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PDF path. Defaults to <input-stem>.<target-lang>.translated.pdf.",
    )
    parser.add_argument(
        "--source-lang",
        default=os.getenv("PAPER_PDF_TRANSLATOR_SOURCE_LANG", "en"),
        help="Source language code. Default: en.",
    )
    parser.add_argument(
        "--target-lang",
        default=os.getenv("PAPER_PDF_TRANSLATOR_TARGET_LANG", "zh"),
        help="Target language code. Default: zh.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "qwen3_235b"),
        help="OpenAI-compatible model name.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://10.100.36.33:8000/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI-compatible API key.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_env_float("OPENAI_TIMEOUT"),
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--qps",
        type=int,
        default=_env_int("PAPER_PDF_TRANSLATOR_QPS", 4),
        help="Translation requests per second limit. Default: 4.",
    )
    parser.add_argument(
        "--pool-max-workers",
        type=int,
        help="Maximum translation worker count.",
    )
    parser.add_argument(
        "--pages",
        help="Optional page selector, e.g. 1-5 or 1,3,8-10.",
    )
    parser.add_argument(
        "--primary-font-family",
        choices=["serif", "sans-serif", "script"],
        help="Override translated text font family.",
    )
    parser.add_argument(
        "--translate-table-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to translate text inside tables. Default: true.",
    )
    parser.add_argument(
        "--auto-ocr",
        action="store_true",
        help="Enable upstream auto OCR workaround for scanned PDFs.",
    )
    parser.add_argument(
        "--compatibility-mode",
        action="store_true",
        help="Enable upstream compatibility-oriented settings.",
    )
    parser.add_argument(
        "--custom-system-prompt",
        help="Optional custom system prompt passed to the translation backend.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional temperature to send to the translation backend.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Optional reasoning effort for compatible backends.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Keep upstream intermediate outputs in this directory instead of a temp dir.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and upstream debug mode.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and the final output path.",
    )
    return parser


def configure_logging(debug: bool, quiet: bool) -> None:
    level = logging.INFO
    if quiet:
        level = logging.WARNING
    elif debug:
        level = logging.DEBUG

    logging.basicConfig(level=level, format="%(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(debug=args.debug, quiet=args.quiet)

    input_pdf = args.input_pdf.expanduser().resolve()
    if args.output and args.output_pdf:
        parser.error("Use either positional output_pdf or --output, not both.")

    output_arg = args.output or args.output_pdf
    output = (
        output_arg.expanduser().resolve()
        if output_arg
        else _default_output_path(input_pdf, args.target_lang)
    )

    request = TranslateRequest(
        input_path=input_pdf,
        output_path=output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        qps=args.qps,
        pool_max_workers=args.pool_max_workers,
        pages=args.pages,
        primary_font_family=args.primary_font_family,
        translate_table_text=args.translate_table_text,
        auto_ocr=args.auto_ocr,
        compatibility_mode=args.compatibility_mode,
        custom_system_prompt=args.custom_system_prompt,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
        work_dir=args.work_dir,
    )

    try:
        output_path = translate_pdf(request)
    except Exception as exc:
        logging.getLogger(__name__).error("Translation failed: %s", exc)
        return 1

    print(output_path)
    return 0
