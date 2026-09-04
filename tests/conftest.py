from pathlib import Path

import pytest

from slack_db_bot.database import ensure_database

DATABASE_URL = (
    "https://github.com/langchain-ai/applied-ai-take-home-database/raw/"
    "4bac5955b1997be7fe2d4c54a09d1aece57d43e1/synthetic_startup.sqlite"
)
DATABASE_HASH = "5bd743daf068f55599e0b93f97f65973298c7123c9d67518f533bd0aa2925c2a"


@pytest.fixture(scope="session")
def database_path() -> Path:
    return ensure_database(Path(".data/synthetic_startup.sqlite"), DATABASE_URL, DATABASE_HASH)
