"""Central configuration for the JEE Advanced Math Agent."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Application configuration with environment variable overrides."""

    LLM_API_KEY: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    LLM_BASE_URL: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    LLM_MODEL: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    TEMPERATURE: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1"))
    )
    MAX_RETRIES: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    MAX_PROBLEMS: int = field(
        default_factory=lambda: int(os.getenv("MAX_PROBLEMS", "0"))
    )

    @property
    def DATA_DIR(self) -> str:
        """Return the absolute path to the data directory."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    def validate(self) -> None:
        """Validate configuration and ensure data directory exists."""
        if not self.LLM_API_KEY:
            raise ValueError(
                "LLM_API_KEY must be set. Provide it via the sidebar or LLM_API_KEY environment variable."
            )
        os.makedirs(self.DATA_DIR, exist_ok=True)
