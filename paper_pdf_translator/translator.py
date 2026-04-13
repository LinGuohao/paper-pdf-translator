from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranslateRequest:
    input_path: Path
    output_path: Path
    source_lang: str = "en"
    target_lang: str = "zh"
    model: str = "qwen3_235b"
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    qps: int = 4
    pool_max_workers: int | None = None
    pages: str | None = None
    primary_font_family: str | None = None
    translate_table_text: bool = True
    auto_ocr: bool = False
    compatibility_mode: bool = False
    custom_system_prompt: str | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    debug: bool = False
    work_dir: Path | None = None

    def validate(self) -> None:
        self.input_path = self.input_path.expanduser().resolve()
        self.output_path = self.output_path.expanduser().resolve()
        if self.work_dir is not None:
            self.work_dir = self.work_dir.expanduser().resolve()

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input PDF does not exist: {self.input_path}")
        if self.input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {self.input_path}")
        if not self.api_key:
            raise ValueError("An API key is required. Pass --api-key or set OPENAI_API_KEY.")
        if not self.base_url:
            raise ValueError(
                "A base URL is required. Pass --base-url or set OPENAI_BASE_URL."
            )
        if self.qps < 1:
            raise ValueError("--qps must be greater than 0.")
        if self.pool_max_workers is not None and self.pool_max_workers < 1:
            raise ValueError("--pool-max-workers must be greater than 0 when provided.")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


def _build_settings(request: TranslateRequest, output_dir: Path):
    from pdf2zh_next.config.model import BasicSettings
    from pdf2zh_next.config.model import PDFSettings
    from pdf2zh_next.config.model import SettingsModel
    from pdf2zh_next.config.model import TranslationSettings
    from pdf2zh_next.config.translate_engine_model import OpenAICompatibleSettings

    translate_engine_settings = OpenAICompatibleSettings(
        openai_compatible_model=request.model,
        openai_compatible_base_url=request.base_url,
        openai_compatible_api_key=request.api_key,
        openai_compatible_timeout=(
            str(request.timeout) if request.timeout is not None else None
        ),
        openai_compatible_temperature=(
            str(request.temperature) if request.temperature is not None else None
        ),
        openai_compatible_reasoning_effort=request.reasoning_effort,
        openai_compatible_send_temperature=request.temperature is not None,
        openai_compatible_send_reasoning_effort=request.reasoning_effort is not None,
    )

    settings = SettingsModel(
        report_interval=0.5,
        basic=BasicSettings(
            input_files=set(),
            debug=request.debug,
        ),
        translation=TranslationSettings(
            lang_in=request.source_lang,
            lang_out=request.target_lang,
            output=str(output_dir),
            qps=request.qps,
            pool_max_workers=request.pool_max_workers,
            custom_system_prompt=request.custom_system_prompt,
            no_auto_extract_glossary=True,
            primary_font_family=request.primary_font_family,
        ),
        pdf=PDFSettings(
            pages=request.pages,
            no_dual=True,
            no_mono=False,
            watermark_output_mode="no_watermark",
            translate_table_text=request.translate_table_text,
            auto_enable_ocr_workaround=request.auto_ocr,
            enhance_compatibility=request.compatibility_mode,
        ),
        translate_engine_settings=translate_engine_settings,
        term_extraction_engine_settings=None,
    )
    settings.validate_settings()
    return settings


def _pick_output_pdf(translate_result) -> Path:
    candidates = [
        getattr(translate_result, "no_watermark_mono_pdf_path", None),
        getattr(translate_result, "mono_pdf_path", None),
        getattr(translate_result, "no_watermark_dual_pdf_path", None),
        getattr(translate_result, "dual_pdf_path", None),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    raise RuntimeError("Translation finished but no translated PDF file was produced.")


def _log_event(event: dict, last_progress_bucket: int) -> int:
    event_type = event.get("type")
    if event_type == "progress_start":
        stage = event.get("stage", "Unknown stage")
        logger.info("Stage started: %s", stage)
    elif event_type == "progress_end":
        stage = event.get("stage", "Unknown stage")
        logger.info("Stage finished: %s", stage)
    elif event_type == "progress_update":
        overall = event.get("overall_progress")
        if overall is not None:
            bucket = int(overall // 10)
            if bucket > last_progress_bucket:
                logger.info(
                    "Progress: %.1f%% (%s)",
                    overall,
                    event.get("stage", "processing"),
                )
                return bucket
    elif event_type == "stage_summary" and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Stage summary: %s", event)
    return last_progress_bucket


async def translate_pdf_async(request: TranslateRequest) -> Path:
    request.validate()

    import babeldoc.assets.assets
    from pdf2zh_next.high_level import do_translate_async_stream

    if request.work_dir is not None:
        request.work_dir.mkdir(parents=True, exist_ok=True)
        work_dir_context = contextlib.nullcontext(request.work_dir)
    else:
        work_dir_context = tempfile.TemporaryDirectory(prefix="paper-pdf-translator-")

    with work_dir_context as raw_work_dir:
        work_dir = Path(raw_work_dir)
        logger.info("Preparing layout assets...")
        babeldoc.assets.assets.warmup()

        settings = _build_settings(request, work_dir)
        last_progress_bucket = -1
        translate_result = None

        async for event in do_translate_async_stream(settings, request.input_path):
            last_progress_bucket = _log_event(event, last_progress_bucket)
            if event.get("type") == "error":
                raise RuntimeError(event.get("error", "Translation failed"))
            if event.get("type") == "finish":
                translate_result = event["translate_result"]
                break

        if translate_result is None:
            raise RuntimeError("Translation did not return a finish event.")

        produced_pdf = _pick_output_pdf(translate_result)
        shutil.copy2(produced_pdf, request.output_path)
        logger.info("Saved translated PDF to %s", request.output_path)
        return request.output_path


def translate_pdf(request: TranslateRequest) -> Path:
    try:
        return asyncio.run(translate_pdf_async(request))
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" not in str(exc):
            raise
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(translate_pdf_async(request))
