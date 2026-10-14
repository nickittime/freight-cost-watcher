from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", extra="ignore", frozen=True
    )

    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    CACHE_DIR: Path = BASE_DIR / ".cache"
    LOG_LEVEL: str = "INFO"

    TRAILING_HISTORY_WEEKS: int = 8
    VS_OWN_SPIKE_THRESHOLD_PCT: float = 10.0
    VS_PEER_CREEP_THRESHOLD_PCT: float = 20.0
    MIN_OWN_RISE_FOR_PEER_CREEP_PCT: float = 5.0
    HYSTERESIS_KEEP_PEER_PCT: float = 15.0

    MAX_WEEKS_OPEN_ENDED_EFFECT: int = 8
    MAGNITUDE_TOLERANCE_PCT: float = 3.0
    ALL_ROUTES_MAX_PEER_GAP_PCT: float = 10.0
    RETRIEVAL_TOP_K: int = 5

    LLM_PROVIDER: Literal["groq", "gemini", "stub"] = "groq"
    GROQ_API_KEY: SecretStr | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_API_KEY: SecretStr | None = None
    GEMINI_MODEL: str = "gemini-3.8-flash"
    LLM_TEMPERATURE: float = 0.0
    LLM_SEED: int = 42
    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT_S: float = 30.0
    PRICE_INPUT_PER_MTOK: float = 0.59
    PRICE_OUTPUT_PER_MTOK: float = 0.79

    @property
    def shipments_path(self) -> Path:
        return self.DATA_DIR / "shipment_records.csv"

    @property
    def notes_path(self) -> Path:
        return self.DATA_DIR / "context_notes.csv"

    @property
    def sample_output_path(self) -> Path:
        return self.DATA_DIR / "sample_output_format_v2.csv"

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.HYSTERESIS_KEEP_PEER_PCT >= self.VS_PEER_CREEP_THRESHOLD_PCT:
            raise ValueError("HYSTERESIS_KEEP_PEER_PCT must be below VS_PEER_CREEP_THRESHOLD_PCT")
        if self.TRAILING_HISTORY_WEEKS < 1:
            raise ValueError("TRAILING_HISTORY_WEEKS must be >= 1")
        if self.LLM_PROVIDER == "groq" and not (
            self.GROQ_API_KEY and self.GROQ_API_KEY.get_secret_value().strip()
        ):
            raise ValueError("LLM_PROVIDER=groq requires a non-empty GROQ_API_KEY in .env")
        if self.LLM_PROVIDER == "gemini" and not (
            self.GEMINI_API_KEY and self.GEMINI_API_KEY.get_secret_value().strip()
        ):
            raise ValueError("LLM_PROVIDER=gemini requires a non-empty GEMINI_API_KEY in .env")
        return self

    def ensure_dirs(self) -> None:
        for d in (self.OUTPUT_DIR, self.CACHE_DIR):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()