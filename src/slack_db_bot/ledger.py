from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path


class EventLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self.connect()) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    thread_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    error_code TEXT
                );
                CREATE TABLE IF NOT EXISTS engaged_threads (
                    thread_key TEXT PRIMARY KEY,
                    engaged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_key TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS thread_summaries (
                    thread_key TEXT PRIMARY KEY,
                    text TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deliveries (
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
                    channel_id TEXT NOT NULL,
                    message_ts TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    delivered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS run_metrics (
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
                    tool_calls INTEGER NOT NULL,
                    model_calls INTEGER NOT NULL,
                    rewrites INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            metric_columns = {row[1] for row in connection.execute("PRAGMA table_info(run_metrics)")}
            for column in ("input_tokens", "output_tokens"):
                if column not in metric_columns:
                    connection.execute(f"ALTER TABLE run_metrics ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            connection.commit()
        self.prune()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def thread_key(team_id: str, channel_id: str, root_ts: str) -> str:
        return hashlib.sha256(f"{team_id}:{channel_id}:{root_ts}".encode()).hexdigest()

    def claim_event(self, event_id: str, thread_key: str) -> bool:
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, updated_at < datetime('now', '-10 minutes') AS stale FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO events(event_id, thread_key, status) VALUES (?, ?, 'processing')",
                    (event_id, thread_key),
                )
                connection.commit()
                return True
            if row["status"] == "failed" or (row["status"] == "processing" and row["stale"]):
                connection.execute(
                    "UPDATE events SET status='processing', error_code=NULL, "
                    "updated_at=CURRENT_TIMESTAMP WHERE event_id=?",
                    (event_id,),
                )
                connection.commit()
                return True
            connection.rollback()
            return False

    def engage(self, thread_key: str) -> None:
        with closing(self.connect()) as connection:
            connection.execute("INSERT OR IGNORE INTO engaged_threads(thread_key) VALUES (?)", (thread_key,))
            connection.commit()

    def is_engaged(self, thread_key: str) -> bool:
        with closing(self.connect()) as connection:
            return (
                connection.execute("SELECT 1 FROM engaged_threads WHERE thread_key=?", (thread_key,)).fetchone()
                is not None
            )

    def history(self, thread_key: str, limit: int = 12) -> list[dict[str, str]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT role, text FROM turns WHERE thread_key=? ORDER BY id DESC LIMIT ?", (thread_key, limit)
            ).fetchall()
            summary = connection.execute(
                "SELECT text FROM thread_summaries WHERE thread_key=?", (thread_key,)
            ).fetchone()
        history = [{"role": row["role"], "text": row["text"][:2000]} for row in reversed(rows)]
        if summary:
            history.insert(0, {"role": "user", "text": "Earlier thread context (untrusted):\n" + summary["text"]})
        return history

    def append_turns(self, thread_key: str, user_text: str, assistant_text: str) -> None:
        with closing(self.connect()) as connection:
            connection.executemany(
                "INSERT INTO turns(thread_key, role, text) VALUES (?, ?, ?)",
                [(thread_key, "user", user_text), (thread_key, "assistant", assistant_text)],
            )
            older = connection.execute(
                "SELECT id, role, text FROM turns WHERE thread_key=? ORDER BY id DESC LIMIT -1 OFFSET 12",
                (thread_key,),
            ).fetchall()
            if older:
                previous = connection.execute(
                    "SELECT text FROM thread_summaries WHERE thread_key=?", (thread_key,)
                ).fetchone()
                summary = (previous["text"] + "\n") if previous else ""
                summary += "\n".join(f"{row['role']}: {row['text'][:500]}" for row in reversed(older))
                connection.execute(
                    "INSERT OR REPLACE INTO thread_summaries(thread_key, text) VALUES (?, ?)",
                    (thread_key, summary[-4000:]),
                )
                connection.execute("DELETE FROM turns WHERE thread_key=? AND id<=?", (thread_key, older[0]["id"]))
            connection.execute(
                "UPDATE engaged_threads SET engaged_at=CURRENT_TIMESTAMP WHERE thread_key=?", (thread_key,)
            )
            connection.commit()

    def prune(self) -> None:
        """Keep local conversation and deduplication records for thirty days."""
        with closing(self.connect()) as connection:
            old_events = "SELECT event_id FROM events WHERE updated_at < datetime('now', '-30 days')"
            connection.execute(f"DELETE FROM deliveries WHERE event_id IN ({old_events})")  # noqa: S608
            connection.execute(f"DELETE FROM run_metrics WHERE event_id IN ({old_events})")  # noqa: S608
            connection.execute("DELETE FROM events WHERE updated_at < datetime('now', '-30 days')")
            old_threads = "SELECT thread_key FROM engaged_threads WHERE engaged_at < datetime('now', '-30 days')"
            connection.execute(f"DELETE FROM turns WHERE thread_key IN ({old_threads})")  # noqa: S608
            connection.execute(f"DELETE FROM thread_summaries WHERE thread_key IN ({old_threads})")  # noqa: S608
            connection.execute("DELETE FROM engaged_threads WHERE engaged_at < datetime('now', '-30 days')")
            connection.commit()

    def delivery(self, event_id: str) -> dict | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT channel_id, message_ts, text_sha256 FROM deliveries WHERE event_id=?", (event_id,)
            ).fetchone()
        return dict(row) if row else None

    def mark_delivered(self, event_id: str, channel_id: str, message_ts: str, text: str) -> None:
        digest = hashlib.sha256(text.encode()).hexdigest()
        with closing(self.connect()) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO deliveries(event_id, channel_id, message_ts, text_sha256) VALUES (?, ?, ?, ?)",
                (event_id, channel_id, message_ts, digest),
            )
            connection.execute(
                "UPDATE events SET status='delivered', updated_at=CURRENT_TIMESTAMP WHERE event_id=?", (event_id,)
            )
            connection.commit()

    def mark_failed(self, event_id: str, error_code: str) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                "UPDATE events SET status='failed', error_code=?, updated_at=CURRENT_TIMESTAMP WHERE event_id=?",
                (error_code[:100], event_id),
            )
            connection.commit()

    def mark_uncertain(self, event_id: str) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                "UPDATE events SET status='uncertain', updated_at=CURRENT_TIMESTAMP WHERE event_id=?", (event_id,)
            )
            connection.commit()

    def record_metrics(self, event_id: str, metrics: dict[str, int], latency_ms: int) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT OR REPLACE INTO run_metrics(
                       event_id, tool_calls, model_calls, rewrites, latency_ms, input_tokens, output_tokens
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    metrics["tool_calls"],
                    metrics["model_calls"],
                    metrics["rewrites"],
                    latency_ms,
                    metrics.get("input_tokens", 0),
                    metrics.get("output_tokens", 0),
                ),
            )
            connection.commit()
