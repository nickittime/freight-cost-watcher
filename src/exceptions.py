class FreightWatcherError(Exception):
    """Base class for all application errors."""


class ConfigurationError(FreightWatcherError):
    """Invalid or missing configuration."""


class DataValidationError(FreightWatcherError):
    """Input CSV violates the expected schema or contains invalid rows."""


class LLMProviderError(FreightWatcherError):
    """Network, auth, or rate-limit failure after retries."""


class LLMOutputError(FreightWatcherError):
    """LLM returned malformed or schema-invalid output."""


class RetrievalError(FreightWatcherError):
    """Vector store build or query failure."""