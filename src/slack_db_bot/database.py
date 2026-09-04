from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any

from sqlglot import exp, parse

ALLOWED_COLUMNS: dict[str, frozenset[str]] = {
    "company_profile": frozenset(
        {
            "company_id",
            "name",
            "category",
            "headquarters",
            "founding_year",
            "mission",
            "ideal_customer_profile",
            "architecture_summary",
            "compliance_posture",
            "pricing_overview",
            "differentiation",
        }
    ),
    "products": frozenset(
        {
            "product_id",
            "name",
            "category",
            "description",
            "target_persona",
            "pricing_model",
            "deployment_modes_json",
            "core_use_cases_json",
            "features_json",
        }
    ),
    "competitors": frozenset(
        {"competitor_id", "name", "segment", "description", "pricing_position", "strengths_json", "weaknesses_json"}
    ),
    "employees": frozenset(
        {
            "employee_id",
            "full_name",
            "title",
            "department",
            "region",
            "management_level",
            "domain_expertise_json",
            "writing_style",
        }
    ),
    "scenarios": frozenset(
        {
            "scenario_id",
            "created_at",
            "industry",
            "region",
            "company_size_band",
            "primary_product_id",
            "secondary_product_id",
            "primary_competitor_id",
            "trigger_event",
            "pain_point",
            "scenario_summary",
            "status",
        }
    ),
    "customers": frozenset(
        {
            "customer_id",
            "scenario_id",
            "name",
            "industry",
            "subindustry",
            "region",
            "country",
            "size_band",
            "employee_count",
            "annual_revenue_band",
            "crm_stage",
            "tech_stack_summary",
            "account_health",
            "primary_contact_name",
            "notes",
        }
    ),
    "implementations": frozenset(
        {
            "implementation_id",
            "scenario_id",
            "customer_id",
            "product_id",
            "deployment_model",
            "status",
            "kickoff_date",
            "go_live_date",
            "contract_value",
            "scope_summary",
            "success_metrics_json",
            "risks_json",
        }
    ),
    "artifacts": frozenset(
        {
            "artifact_id",
            "scenario_id",
            "customer_id",
            "product_id",
            "competitor_id",
            "artifact_type",
            "title",
            "created_at",
            "summary",
            "content_text",
            "token_estimate",
            "metadata_json",
        }
    ),
    "artifacts_fts": frozenset({"artifact_id", "title", "summary", "content_text", "rank"}),
}

FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Transaction,
    exp.Attach,
    exp.Detach,
    exp.Pragma,
)


class UnsafeQuery(ValueError):
    pass


def ensure_database(path: Path, url: str, expected_sha256: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if not url.startswith("https://"):
            raise ValueError("Database download URL must use HTTPS")
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as target:  # noqa: S310
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            downloaded = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if downloaded != expected_sha256.lower():
                raise ValueError(f"Downloaded database hash mismatch: expected {expected_sha256}, got {downloaded}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256.lower():
        raise ValueError(f"Database hash mismatch for {path}: expected {expected_sha256}, got {actual}")
    return path


def validate_read_query(sql: str, row_limit: int = 100) -> str:
    if len(sql) > 8_000 or "\x00" in sql:
        raise UnsafeQuery("Query is too large or contains invalid bytes")
    try:
        statements = parse(sql, read="sqlite")
    except Exception as exc:
        raise UnsafeQuery("Invalid SQL") from exc
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union)):
        raise UnsafeQuery("Exactly one SELECT query is required")
    statement = statements[0]
    if statement.find(exp.Star) is not None:
        raise UnsafeQuery("SELECT * is not allowed")
    if any(statement.find(node) is not None for node in FORBIDDEN_NODES):
        raise UnsafeQuery("Only read queries are allowed")
    tables = {table.name for table in statement.find_all(exp.Table)}
    if not tables or not tables <= ALLOWED_COLUMNS.keys():
        raise UnsafeQuery("Query references an unapproved table")
    aliases = {table.alias_or_name: table.name for table in statement.find_all(exp.Table)}
    for column in statement.find_all(exp.Column):
        if column.name == "*":
            raise UnsafeQuery("SELECT * is not allowed")
        if column.table:
            table = aliases.get(column.table, column.table)
            if table in ALLOWED_COLUMNS and column.name not in ALLOWED_COLUMNS[table]:
                raise UnsafeQuery(f"Column {column.name!r} is not allowed for {table}")
        elif column.name not in tables and not any(column.name in ALLOWED_COLUMNS[table] for table in tables):
            raise UnsafeQuery(f"Column {column.name!r} is not allowed")
    if len(list(statement.find_all(exp.Join))) > 4:
        raise UnsafeQuery("Query contains too many joins")
    allowed_functions = {
        "count",
        "min",
        "max",
        "sum",
        "avg",
        "lower",
        "upper",
        "length",
        "substr",
        "substring",
        "coalesce",
        "json_extract",
        "date",
        "datetime",
        "strftime",
        "round",
        "abs",
        "bm25",
    }
    for function in statement.find_all(exp.Func):
        name = function.name.lower() if isinstance(function, exp.Anonymous) else function.sql_name().lower()
        if name not in allowed_functions:
            raise UnsafeQuery(f"Function {name!r} is not allowed")
    limit = statement.args.get("limit")
    if limit is None:
        statement.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
    else:
        expression = limit.expression
        if not isinstance(expression, exp.Literal) or not expression.is_int or int(expression.this) > row_limit:
            raise UnsafeQuery(f"LIMIT must be an integer no greater than {row_limit}")
    return statement.sql(dialect="sqlite")


class ReadOnlyDatabase:
    def __init__(self, path: Path, *, row_limit: int = 100, timeout_ms: int = 750):
        self.path = path.resolve()
        self.row_limit = row_limit
        self.timeout_ms = timeout_ms

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"{self.path.as_uri()}?mode=ro&immutable=1", uri=True, timeout=self.timeout_ms / 1000
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        safe_sql = validate_read_query(sql, self.row_limit)
        with closing(self.connect()) as connection:
            deadline = time.monotonic() + self.timeout_ms / 1000

            def progress() -> int:
                return int(time.monotonic() >= deadline)

            connection.set_progress_handler(progress, 1_000)
            try:
                rows = [dict(row) for row in connection.execute(safe_sql, parameters).fetchmany(self.row_limit + 1)]
                if len(rows) > self.row_limit or len(json.dumps(rows)) > 1_000_000:
                    raise UnsafeQuery("Database result exceeds its size budget")
                return rows
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc).lower():
                    raise TimeoutError("Database query exceeded its operation budget") from exc
                raise

    def all_artifacts(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.query(
                """SELECT a.artifact_id, a.artifact_type, a.title, a.created_at, a.summary,
                          a.content_text, a.metadata_json, c.name AS customer_name, c.region, c.country
                   FROM artifacts AS a LEFT JOIN customers AS c ON c.customer_id = a.customer_id
                   ORDER BY a.artifact_id LIMIT 100 OFFSET ?""",
                (offset,),
            )
            rows.extend(page)
            if len(page) < 100:
                break
            offset += 100
        return rows


def terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
