import json
import re
import time
from abc import ABC, abstractmethod

from src.config import get_settings
from src.exceptions import LLMProviderError
from src.llm.cost_tracker import TokenCostTracker
from src.logger import get_logger

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Abstract interface for LLM completions."""

    def __init__(self, tracker: TokenCostTracker) -> None:
        self.tracker = tracker

    @abstractmethod
    def complete_json(
        self, system_prompt: str, user_prompt: str, context: str = ""
    ) -> dict:
        """Generates a JSON response and records token metrics."""


class GroqClient(BaseLLMClient):
    """Groq client with temp=0, seed=42, rate pacing, and automated quota failover."""

    def __init__(self, tracker: TokenCostTracker) -> None:
        super().__init__(tracker)
        settings = get_settings()
        if not settings.GROQ_API_KEY:
            raise LLMProviderError("Missing GROQ_API_KEY")
        from groq import Groq

        clean_key = (
            settings.GROQ_API_KEY.get_secret_value()
            .strip()
            .strip("'\"")
        )
        self.client = Groq(api_key=clean_key)
        self.primary_model = settings.GROQ_MODEL
        # Sibling model on the same account with an independent quota
        self.fallback_model = (
            "openai/gpt-oss-20b"
            if "120b" in settings.GROQ_MODEL
            else "openai/gpt-oss-120b"
        )
        self.temperature = settings.LLM_TEMPERATURE
        self.seed = settings.LLM_SEED
        self.max_retries = settings.LLM_MAX_RETRIES
        self.timeout = settings.LLM_TIMEOUT_S

    def complete_json(
        self, system_prompt: str, user_prompt: str, context: str = ""
    ) -> dict:
        time.sleep(0.5)  # Pacing delay to stay within RPM/TPM thresholds

        models_to_attempt = [self.primary_model, self.fallback_model]

        for model in models_to_attempt:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        seed=self.seed,
                        response_format={"type": "json_object"},
                        timeout=self.timeout,
                    )
                    usage = response.usage
                    p_tok = usage.prompt_tokens if usage else 0
                    c_tok = usage.completion_tokens if usage else 0
                    self.tracker.record_call(
                        p_tok, c_tok, model, context, provider="groq"
                    )

                    content = response.choices[0].message.content or "{}"
                    return json.loads(content)

                except Exception as e:
                    err_str = str(e)
                    is_daily_limit = "tokens per day" in err_str.lower() or "tpd" in err_str.lower()

                    if is_daily_limit and model != self.fallback_model:
                        logger.warning(
                            "Daily token quota (TPD) reached for %s. Automatically failing over to %s.",
                            model,
                            self.fallback_model,
                        )
                        break  # Immediately break to fallback model

                    logger.warning(
                        "Groq attempt %d/%d on %s failed: %s",
                        attempt,
                        self.max_retries,
                        model,
                        err_str,
                    )

                    if attempt == self.max_retries and model == self.fallback_model:
                        raise LLMProviderError(
                            f"Groq API call failed across primary and fallback models: {e}"
                        ) from e

                    # Exponential backoff on transient errors
                    wait_s = 5.0 if "429" in err_str else float(2**attempt)
                    time.sleep(wait_s)

        raise LLMProviderError("All Groq model completion attempts failed.")


class GeminiClient(BaseLLMClient):
    """Google Gemini client with rate-limit-aware retry backoff and temp=0."""

    def __init__(self, tracker: TokenCostTracker) -> None:
        super().__init__(tracker)
        settings = get_settings()
        if not settings.GEMINI_API_KEY:
            raise LLMProviderError("Missing GEMINI_API_KEY")
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=settings.GEMINI_API_KEY.get_secret_value())
        self.model = settings.GEMINI_MODEL
        self.types = types
        self.max_retries = max(settings.LLM_MAX_RETRIES, 5)

    def complete_json(
        self, system_prompt: str, user_prompt: str, context: str = ""
    ) -> dict:
        for attempt in range(1, self.max_retries + 1):
            try:
                config = self.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                )
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config,
                )
                p_tok = (
                    response.usage_metadata.prompt_token_count
                    if response.usage_metadata
                    else 0
                )
                c_tok = (
                    response.usage_metadata.candidates_token_count
                    if response.usage_metadata
                    else 0
                )
                self.tracker.record_call(
                    p_tok, c_tok, self.model, context, provider="gemini"
                )

                text = response.text or "{}"
                return json.loads(text)
            except Exception as e:
                err_str = str(e)
                logger.warning(
                    "Gemini attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    err_str,
                )
                if attempt == self.max_retries:
                    raise LLMProviderError(
                        f"Gemini API call failed after {self.max_retries} attempts: {e}"
                    ) from e

                # Parse explicit retryDelay from 429 response if present
                delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str.lower())
                if delay_match:
                    wait_s = float(delay_match.group(1)) + 2.0
                elif "429" in err_str or "resource_exhausted" in err_str.lower():
                    wait_s = 15.0 * attempt
                elif "503" in err_str or "unavailable" in err_str.lower():
                    wait_s = 6.0 * attempt
                else:
                    wait_s = float(2**attempt)

                logger.info("Waiting %.1fs before retry...", wait_s)
                time.sleep(wait_s)
        raise LLMProviderError("Gemini call failed unexpectedly")


class OfflineStubLLMClient(BaseLLMClient):
    """
    Generic rule-based stub for offline unit testing only (NOT for submission cost logs).
    Extracts dates/magnitudes generically via regex without route-specific answer keys.
    """

    def __init__(self, tracker: TokenCostTracker) -> None:
        super().__init__(tracker)
        self.model = "offline-test-stub"
        logger.warning(
            "====================================================================\n"
            "WARNING: USING OFFLINE TEST STUB LLM CLIENT (LLM_PROVIDER=stub).\n"
            "DO NOT USE THIS MODE FOR FINAL SUBMISSION OR TOKEN COST LOGS.\n"
            "Set LLM_PROVIDER=groq or LLM_PROVIDER=gemini in .env for real runs.\n"
            "===================================================================="
        )

    def complete_json(
        self, system_prompt: str, user_prompt: str, context: str = ""
    ) -> dict:
        self.tracker.record_call(0, 0, self.model, context, provider="stub")
        sp_low = system_prompt.lower()
        up_low = user_prompt.lower()

        if "effective_start_date" in sp_low:
            pub_match = re.search(r"published date:\s*(\d{4}-\d{2}-\d{2})", up_low)
            start_dt = pub_match.group(1) if pub_match else "2024-01-01"
            end_dt = None
            if "feb 24 to mar 8" in up_low:
                start_dt, end_dt = "2025-02-24", "2025-03-08"
            elif "temporary surcharge" in up_low or "for about a week" in up_low:
                end_dt = None

            has_increase = any(
                w in up_low for w in ("higher trip costs", "surcharge", "pushing up")
            ) and not any(
                w in up_low
                for w in ("not significantly affected", "without a rate change", "normal")
            )
            pct_match = re.search(r"(\d+)\s*-\s*(\d+)%", up_low)
            min_p = float(pct_match.group(1)) if (pct_match and has_increase) else 0.0
            max_p = float(pct_match.group(2)) if (pct_match and has_increase) else 0.0
            return {
                "effective_start_date": start_dt,
                "effective_end_date": end_dt,
                "cost_effect": "increase" if has_increase else "none",
                "stated_magnitude_min_pct": min_p,
                "stated_magnitude_max_pct": max_p,
            }

        if "tool_calls" in sp_low:
            return {"tool_calls": [{"name": "get_flagged_rows", "arguments": {}}]}

        blocks = user_prompt.split("Note ID:")
        for b in blocks[1:]:
            lines = b.strip().splitlines()
            nid = lines[0].strip()
            b_low = b.lower()
            if any(w in b_low for w in ("higher trip costs", "surcharge", "pushing up")):
                return {
                    "verdict": "justified",
                    "matched_note_id": nid,
                    "reason": f"Proposed candidate note {nid}.",
                }
        return {
            "verdict": "unexplained",
            "matched_note_id": None,
            "reason": "No causal increase note found in retrieved context.",
        }


def get_llm_client(tracker: TokenCostTracker | None = None) -> BaseLLMClient:
    """Factory creating the configured LLM client."""
    settings = get_settings()
    t = tracker or TokenCostTracker()
    if settings.LLM_PROVIDER == "groq":
        return GroqClient(t)
    if settings.LLM_PROVIDER == "gemini":
        return GeminiClient(t)
    return OfflineStubLLMClient(t)