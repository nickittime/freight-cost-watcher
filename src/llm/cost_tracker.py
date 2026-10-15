import json
from datetime import UTC, datetime
from pathlib import Path

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)


class TokenCostTracker:
    """Tracks token usage, provider, model, and USD cost for all LLM calls in a run."""

    def __init__(
        self,
        price_input_per_mtok: float | None = None,
        price_output_per_mtok: float | None = None,
    ) -> None:
        settings = get_settings()
        self.provider: str = settings.LLM_PROVIDER
        self.model: str = settings.GROQ_MODEL
        self.price_input = (
            price_input_per_mtok if price_input_per_mtok is not None else settings.PRICE_INPUT_PER_MTOK
        )
        self.price_output = (
            price_output_per_mtok if price_output_per_mtok is not None else settings.PRICE_OUTPUT_PER_MTOK
        )
        self.total_requests: int = 0
        self.total_retries: int = 0
        self.total_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.records: list[dict] = []

    def record_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        context: str = "",
        provider: str | None = None,
        retries: int = 0,
    ) -> None:
        self.total_requests += 1
        self.total_retries += retries
        self.total_calls += 1 + retries
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        if provider:
            self.provider = provider
        self.model = model
        self.records.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "provider": self.provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "retries": retries,
            "context": context,
        })

    def summary(self) -> dict:
        in_cost = (self.total_input_tokens / 1_000_000.0) * self.price_input
        out_cost = (self.total_output_tokens / 1_000_000.0) * self.price_output
        return {
            "provider": self.provider,
            "model": self.model,
            "unique_requests": self.total_requests,
            "retry_attempts": self.total_retries,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "price_input_per_mtok": self.price_input,
            "price_output_per_mtok": self.price_output,
            "input_cost_usd": round(in_cost, 6),
            "output_cost_usd": round(out_cost, 6),
            "total_cost_usd": round(in_cost + out_cost, 6),
        }

    def save_log(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "call_records": self.records,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved token and cost log to %s", output_path)