from pathlib import Path

import pytest
from conftest import DATABASE_HASH, DATABASE_URL

from slack_db_bot.database import ReadOnlyDatabase, UnsafeQuery, ensure_database, validate_read_query


def test_database_hash_and_read_only_connection(database_path: Path) -> None:
    assert ensure_database(database_path, DATABASE_URL, DATABASE_HASH) == database_path
    database = ReadOnlyDatabase(database_path)
    assert database.query("SELECT name FROM customers WHERE name = ? LIMIT 1", ("BlueHarbor Logistics",)) == [
        {"name": "BlueHarbor Logistics"}
    ]
    with pytest.raises(Exception, match=r"readonly|authorized|not authorized"):
        database.connect().execute("DELETE FROM customers")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "SELECT name FROM customers; DROP TABLE customers",
        "PRAGMA database_list",
        "SELECT email FROM employees",
        "SELECT * FROM customers",
        "SELECT name FROM sqlite_master",
        "SELECT a.name FROM customers a JOIN customers b ON 1 JOIN customers c ON 1 "
        "JOIN customers d ON 1 JOIN customers e ON 1 JOIN customers f ON 1",
        "SELECT name FROM customers LIMIT 101",
    ],
)
def test_sql_guard_rejects_unsafe_queries(sql: str) -> None:
    with pytest.raises(UnsafeQuery):
        validate_read_query(sql, row_limit=100)


def test_sql_guard_adds_bounded_limit_and_allows_fts() -> None:
    assert validate_read_query("SELECT name FROM customers") == "SELECT name FROM customers LIMIT 100"
    safe = validate_read_query(
        "SELECT artifact_id, bm25(artifacts_fts) AS rank FROM artifacts_fts "
        "WHERE artifacts_fts MATCH ? ORDER BY rank LIMIT 20"
    )
    assert "LIMIT 20" in safe
