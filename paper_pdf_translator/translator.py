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
    model: str = "qwen3.5-35b-a3b"
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
        openai_compatible_api_key=request.api_key or "EMPTY",
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
            # Force in-process translation so local monkey patches apply reliably.
            debug=True,
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


def _patch_pdf2zh_next_runtime() -> None:
    import importlib

    import babeldoc.format.pdf.high_level as babeldoc_high_level
    import httpx
    import openai
    import requests
    from tenacity import before_sleep_log
    from tenacity import retry
    from tenacity import retry_if_exception
    from tenacity import stop_after_attempt
    from tenacity import wait_exponential

    import pdf2zh_next.translator.utils as translator_utils
    from pdf2zh_next.translator.translator_impl.openai import OpenAITranslator

    if getattr(translator_utils, "_paper_pdf_translator_patched", False):
        return

    original_openai_init = OpenAITranslator.__init__

    class _RetryableOpenAICompatibleError(Exception):
        pass

    def _should_retry_openai_error(exc: BaseException) -> bool:
        if isinstance(
            exc,
            (
                _RetryableOpenAICompatibleError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
            ),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            status_code = getattr(exc, "status_code", None)
            return status_code == 429 or (status_code is not None and status_code >= 500)
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response is not None and exc.response.status_code >= 500
        return False

    def _update_usage_counters(translator, response) -> None:
        try:
            usage = None
            if isinstance(response, dict):
                usage = response.get("usage")
            elif hasattr(response, "usage") and response.usage:
                usage = response.usage
            if not usage:
                return

            if isinstance(usage, dict):
                total_tokens = usage.get("total_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                prompt_cache_hit_tokens = usage.get("prompt_cache_hit_tokens")
                prompt_token_details = usage.get("prompt_tokens_details") or {}
                cached_tokens = prompt_token_details.get("cached_tokens")
            else:
                total_tokens = getattr(usage, "total_tokens", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                prompt_cache_hit_tokens = getattr(
                    usage, "prompt_cache_hit_tokens", None
                )
                prompt_token_details = getattr(usage, "prompt_tokens_details", None)
                cached_tokens = getattr(prompt_token_details, "cached_tokens", None)

            if total_tokens is not None:
                translator.token_count.inc(total_tokens)
            if prompt_tokens is not None:
                translator.prompt_token_count.inc(prompt_tokens)
            if completion_tokens is not None:
                translator.completion_token_count.inc(completion_tokens)
            if prompt_cache_hit_tokens is not None:
                translator.cache_hit_prompt_token_count.inc(prompt_cache_hit_tokens)
            elif cached_tokens is not None:
                translator.cache_hit_prompt_token_count.inc(cached_tokens)
        except Exception as exc:  # pragma: no cover - best effort bookkeeping
            logger.debug("Failed to record token usage: %s", exc)

    def _minimal_chat_completion_request(self, messages, rate_limit_params=None):
        base_url = getattr(self, "_paper_pdf_translator_base_url", None)
        if not base_url:
            raise RuntimeError("Missing OpenAI-compatible base URL.")

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = getattr(self, "_paper_pdf_translator_api_key", None)
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"bearer {api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.send_temperature and self.temperature:
            payload["temperature"] = float(self.temperature)
        if self.send_reasoning_effort and self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        timeout = getattr(self, "_paper_pdf_translator_timeout", None) or 120
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code >= 500:
            raise _RetryableOpenAICompatibleError(
                f"OpenAI-compatible backend returned {response.status_code}: "
                f"{response.text[:500]}"
            )
        if response.status_code == 429:
            raise _RetryableOpenAICompatibleError(
                f"OpenAI-compatible backend returned 429: {response.text[:500]}"
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAI-compatible request failed with status {response.status_code}: "
                f"{response.text[:1000]}"
            )
        return response.json()

    def _patched_openai_init(self, settings, rate_limiter):
        original_openai_init(self, settings, rate_limiter)
        self._paper_pdf_translator_base_url = (
            settings.translate_engine_settings.openai_base_url
        )
        self._paper_pdf_translator_api_key = (
            settings.translate_engine_settings.openai_api_key
        )
        timeout = settings.translate_engine_settings.openai_timeout
        self._paper_pdf_translator_timeout = float(timeout) if timeout else 120.0

    @retry(
        retry=retry_if_exception(_should_retry_openai_error),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _patched_do_translate(self, text, rate_limit_params: dict = None) -> str:
        response = _minimal_chat_completion_request(
            self,
            messages=self.prompt(text),
            rate_limit_params=rate_limit_params,
        )
        _update_usage_counters(self, response)
        message = response["choices"][0]["message"]["content"].strip()
        return self._remove_cot_content(message)

    @retry(
        retry=retry_if_exception(_should_retry_openai_error),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _patched_do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        response = _minimal_chat_completion_request(
            self,
            messages=[{"role": "user", "content": text}],
            rate_limit_params=rate_limit_params,
        )
        _update_usage_counters(self, response)
        message = response["choices"][0]["message"]["content"].strip()
        return self._remove_cot_content(message)

    def _create_translator_instance_without_health_check(
        settings,
        translator_config,
        rate_limiter,
        enforce_glossary_support: bool = True,
    ):
        if isinstance(
            translator_config,
            translator_utils.NOT_SUPPORTED_TRANSLATION_ENGINE_SETTING_TYPE,
        ):
            raise translator_utils.TranslateEngineSettingError(
                f"{translator_config.translate_engine_type} is not supported, Please use other translator!"
            )

        for metadata in translator_utils.TRANSLATION_ENGINE_METADATA:
            if isinstance(translator_config, metadata.setting_model_type):
                translate_engine_type = metadata.translate_engine_type
                logger.info("Using %s translator", translate_engine_type)
                model_name = (
                    f"pdf2zh_next.translator.translator_impl.{translate_engine_type.lower()}"
                )
                module = importlib.import_module(model_name)

                if (
                    enforce_glossary_support
                    and settings.translation.glossaries
                    and not metadata.support_llm
                ):
                    raise translator_utils.TranslateEngineSettingError(
                        f"{translate_engine_type} does not support glossary. Please choose a different translator or remove the glossary."
                    )

                temp_settings = settings.model_copy()
                temp_settings.translate_engine_settings = translator_config
                translator = getattr(module, f"{translate_engine_type}Translator")(
                    temp_settings, rate_limiter
                )

                recommended_qps = getattr(
                    translator, "pdf2zh_next_recommended_qps", None
                )
                recommended_pool_max_workers = getattr(
                    translator, "pdf2zh_next_recommended_pool_max_workers", None
                )
                return translator, recommended_qps, recommended_pool_max_workers

        raise ValueError("No translator found")

    OpenAITranslator.__init__ = _patched_openai_init
    OpenAITranslator.do_translate = _patched_do_translate
    OpenAITranslator.do_llm_translate = _patched_do_llm_translate
    translator_utils._create_translator_instance = (
        _create_translator_instance_without_health_check
    )
    # Force BabelDOC to use the simpler per-paragraph translation path.
    # The LLM-only batch JSON path is more fragile with the current backend.
    babeldoc_high_level.translator_supports_llm = lambda _translator: False
    translator_utils._paper_pdf_translator_patched = True


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

    try:
        import babeldoc.assets.assets
        from pdf2zh_next.high_level import do_translate_async_stream
    except ModuleNotFoundError as exc:
        missing = exc.name or "required dependency"
        raise RuntimeError(
            "Missing runtime dependency "
            f"'{missing}'. Install project dependencies first with: "
            "`python -m pip install -e .`"
        ) from exc

    _patch_pdf2zh_next_runtime()

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
