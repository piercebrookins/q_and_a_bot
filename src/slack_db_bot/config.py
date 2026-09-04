from __future__ import annotations

from functools import cached_property
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    slack_bot_token: SecretStr | None = None
    slack_app_token: SecretStr | None = None
    slack_allowed_workspace_id: str = ""
    slack_allowed_channel_ids: str = ""
    slack_test_channel_id: str = ""

    openai_api_key: SecretStr | None = None
    openai_router_model: str = "gpt-5.4-mini"
    openai_synthesis_model: str = "gpt-5.4-mini"
    openai_eval_model: str = "gpt-5.4-mini"

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "langchain-applied-ai-takehome"
    langsmith_workspace_id: str = ""
    langsmith_hide_inputs: bool = True
    langsmith_hide_outputs: bool = True

    database_url: str = (
        "https://github.com/langchain-ai/applied-ai-take-home-database/raw/"
        "4bac5955b1997be7fe2d4c54a09d1aece57d43e1/synthetic_startup.sqlite"
    )
    database_sha256: str = "5bd743daf068f55599e0b93f97f65973298c7123c9d67518f533bd0aa2925c2a"
    database_path: Path = Path(".data/synthetic_startup.sqlite")
    checkpoint_database_path: Path = Path(".data/checkpoints.sqlite")
    event_ledger_database_path: Path = Path(".data/events.sqlite")
    semantic_index_path: Path = Path(".data/semantic-index")
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"

    log_level: str = "INFO"
    max_tool_calls: int = Field(default=8, ge=1, le=20)
    max_query_rewrites: int = Field(default=2, ge=0, le=4)
    max_model_calls: int = Field(default=6, ge=2, le=8)
    max_slack_calls: int = Field(default=6, ge=4, le=10)
    turn_timeout_seconds: float = Field(default=120, gt=0, le=300)
    max_output_tokens: int = Field(default=4096, ge=256, le=8192)
    max_pending_turns: int = Field(default=32, ge=1, le=100)
    max_concurrent_turns: int = Field(default=4, ge=1, le=16)
    max_evidence: int = Field(default=20, ge=2, le=30)
    sql_row_limit: int = Field(default=100, ge=1, le=500)
    sql_timeout_ms: int = Field(default=750, ge=50, le=5000)
    run_live_tests: bool = False

    @field_validator("database_path", mode="before")
    @classmethod
    def blank_database_path_uses_default(cls, value: object) -> object:
        return value or Path(".data/synthetic_startup.sqlite")

    @field_validator("openai_router_model", mode="before")
    @classmethod
    def blank_router_model_uses_default(cls, value: object) -> object:
        return value or "gpt-5.4-mini"

    @field_validator("openai_synthesis_model", mode="before")
    @classmethod
    def blank_synthesis_model_uses_default(cls, value: object) -> object:
        return value or "gpt-5.4-mini"

    @field_validator("openai_eval_model", mode="before")
    @classmethod
    def blank_eval_model_uses_default(cls, value: object) -> object:
        return value or "gpt-5.4-mini"

    @field_validator("local_embedding_model", mode="before")
    @classmethod
    def blank_embedding_model_uses_default(cls, value: object) -> object:
        return value or "BAAI/bge-small-en-v1.5"

    @cached_property
    def allowed_channels(self) -> frozenset[str]:
        return frozenset(x.strip() for x in self.slack_allowed_channel_ids.split(",") if x.strip())

    @model_validator(mode="after")
    def validate_hash(self) -> Settings:
        if len(self.database_sha256) != 64:
            raise ValueError("DATABASE_SHA256 must be a 64-character SHA-256 digest")
        return self

    def require_live(self) -> Settings:
        missing = []
        for name in (
            "slack_bot_token",
            "slack_app_token",
            "openai_api_key",
            "slack_allowed_workspace_id",
            "slack_allowed_channel_ids",
        ):
            if not getattr(self, name):
                missing.append(name.upper())
        if missing:
            raise ValueError(f"Missing required live settings: {', '.join(missing)}")
        if self.slack_bot_token is None or self.slack_app_token is None:
            raise ValueError("Slack tokens are required")
        if not self.slack_bot_token.get_secret_value().startswith("xoxb-"):
            raise ValueError("SLACK_BOT_TOKEN must start with xoxb-")
        if not self.slack_app_token.get_secret_value().startswith("xapp-"):
            raise ValueError("SLACK_APP_TOKEN must start with xapp-")
        return self
