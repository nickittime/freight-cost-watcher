# src/llm/client.py
import json
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
    """Groq client using llama-3.3-70b-versatile with temp=0 and seed=42."""

    def __init__(self, tracker: TokenCostTracker) -> None:
        super().__init__(tracker)
        settings = get_settings()
        if not settings.GROQ_API_KEY:
            raise LLMProviderError("Missing GROQ_API_KEY")
        from groq import Groq

        self.client = Groq(api_key=settings.GROQ_API_KEY.get_secret_value())
        self.model = settings.GROQ_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.seed = settings.LLM_SEED
        self.max_retries = settings.LLM_MAX_RETRIES
        self.timeout = settings.LLM_TIMEOUT_S

    def complete_json(
        self, system_prompt: str, user_prompt: str, context: str = ""
    ) -> dict:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
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
                self.tracker.record_call(p_tok, c_tok, self.model, context)

                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:
                logger.warning(
                    "Groq attempt %d/%d failed: %s", attempt, self.max_retries, e
                )
                if attempt == self.max_retries:
                    raise LLMProviderError(
                        f"Groq API call failed after {self.max_retries} attempts: {e}"
                    ) from e
                time.sleep(2**attempt)
        raise LLMProviderError("Groq call failed unexpectedly")


class GeminiClient(BaseLLMClient):
    """Google Gemini client using gemini-2.5-flash with temp=0."""

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
        self.max_retries = settings.LLM_MAX_RETRIES

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
                self.tracker.record_call(p_tok, c_tok, self.model, context)

                text = response.text or "{}"
                return json.loads(text)
            except Exception as e:
                logger.warning(
                    "Gemini attempt %d/%d failed: %s", attempt, self.max_retries, e
                )
                if attempt == self.max_retries:
                    raise LLMProviderError(
                        f"Gemini API call failed after {self.max_retries} attempts: {e}"
                    ) from e
                time.sleep(2**attempt)
        raise LLMProviderError("Gemini call failed unexpectedly")


class FakeLLMClient(BaseLLMClient):
    """
    Deterministic offline client for fast, zero-credential tests and evaluations.
    Emits token usage and ground-truth responses without network access.
    """

    def __init__(self, tracker: TokenCostTracker) -> None:
        super().__init__(tracker)
        self.model = "fake-eval-model"

    def complete_json(
        self, system_prompt: str, user_prompt: str, context: str = ""
    ) -> dict:
        self.tracker.record_call(280, 60, self.model, context)

        # Context note enrichment handler
        if "effective_start_date" in system_prompt or "enrich" in system_prompt.lower():
            if "N001" in user_prompt:
                return {
                    "effective_start_date": "2025-02-24",
                    "effective_end_date": "2025-03-08",
                    "cost_effect": "increase",
                    "stated_magnitude_min_pct": 0.0,
                    "stated_magnitude_max_pct": 100.0,
                }
            if "N002" in user_prompt:
                return {
                    "effective_start_date": "2025-01-20",
                    "effective_end_date": "2025-01-26",
                    "cost_effect": "increase",
                    "stated_magnitude_min_pct": 0.0,
                    "stated_magnitude_max_pct": 100.0,
                }
            if "N003" in user_prompt:
                return {
                    "effective_start_date": "2025-05-05",
                    "effective_end_date": None,
                    "cost_effect": "increase",
                    "stated_magnitude_min_pct": 5.0,
                    "stated_magnitude_max_pct": 7.0,
                }
            return {
                "effective_start_date": "2024-01-01",
                "effective_end_date": None,
                "cost_effect": "none",
                "stated_magnitude_min_pct": 0.0,
                "stated_magnitude_max_pct": 0.0,
            }

        # Verifier handler
        if "Ahmedabad-Mumbai" in user_prompt and "2025-01-20" in user_prompt:
            return {
                "verdict": "justified",
                "matched_note_id": "N002",
                "reason": (
                    "Matches note N002 dated 2025-01-20: a regional festival week drove a "
                    "temporary surcharge on this corridor. The cost rise has a clear explanation."
                ),
            }
        if "Chennai-Bangalore" in user_prompt and (
            "2025-02-24" in user_prompt or "2025-03-03" in user_prompt
        ):
            return {
                "verdict": "justified",
                "matched_note_id": "N001",
                "reason": (
                    "Matches note N001 dated 2025-02-24: heavy flooding disrupted normal truck movement "
                    "from Feb 24 to Mar 8, forcing longer detours and higher trip costs."
                ),
            }
        if "Chennai-Bangalore" in user_prompt and (
            "2025-03-10" in user_prompt or "2025-03-17" in user_prompt
        ):
            return {
                "verdict": "unexplained",
                "matched_note_id": None,
                "reason": (
                    "Note N001 confirms flooding ended by Mar 8 and roads were passable by Mar 9, "
                    "while N009 confirms returned conditions. Lingering cost rise is unexplained."
                ),
            }
        if "Mumbai-Pune" in user_prompt and "2025-09-15" in user_prompt:
            return {
                "verdict": "unexplained",
                "matched_note_id": None,
                "reason": (
                    "The closest note (N006, 2025-09-22) mentions stable demand with no major "
                    "disruptions -- it does not describe a reason for a cost rise on this route. "
                    "No genuine justification found; flagged for review."
                ),
            }

        return {
            "verdict": "unexplained",
            "matched_note_id": None,
            "reason": (
                "No matching note found for this route or date range. Cost rise looks unexplained "
                "and worth a human review."
            ),
        }


def get_llm_client(tracker: TokenCostTracker | None = None) -> BaseLLMClient:
    """Factory creating the configured LLM client."""
    settings = get_settings()
    t = tracker or TokenCostTracker()
    if settings.LLM_PROVIDER == "groq":
        return GroqClient(t)
    if settings.LLM_PROVIDER == "gemini":
        return GeminiClient(t)
    return FakeLLMClient(t)