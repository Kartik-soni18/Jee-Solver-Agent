"""Central configuration for the JEE Advanced Math Agent.

Reads TOGETHER_API_KEY from environment or a local .env file.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


@dataclass
class Config:
    """Application configuration — Together AI only."""

    LLM_API_KEY: str = field(default_factory=lambda: os.getenv("TOGETHER_API_KEY", ""))
    LLM_BASE_URL: str = "https://api.together.xyz/v1"
    LLM_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    TEMPERATURE: float = 0.1
    MAX_RETRIES: int = 3

    @property
    def DATA_DIR(self) -> str:
        """Return the absolute path to the data directory."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    def validate(self) -> None:
        """Validate configuration and ensure data directory exists."""
        if not self.LLM_API_KEY:
            raise ValueError(
                "TOGETHER_API_KEY environment variable must be set."
            )
        os.makedirs(self.DATA_DIR, exist_ok=True)
