from __future__ import annotations

import argparse
import asyncio
import logging
import os

import aiosqlite
import structlog
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .answering import GroundedGraph
from .config import Settings
from .database import ReadOnlyDatabase, ensure_database
from .ledger import EventLedger
from .retrieval import HybridRetriever
from .slack_app import build_socket_app


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(message)s")
    for name in ("httpx", "httpcore", "openai", "slack_sdk", "slack_bolt"):
        logging.getLogger(name).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
    )


def configure_langsmith_environment(settings: Settings) -> None:
    values = {
        "LANGSMITH_TRACING": str(settings.langsmith_tracing).lower(),
        "LANGSMITH_ENDPOINT": settings.langsmith_endpoint,
        "LANGSMITH_PROJECT": settings.langsmith_project,
        "LANGSMITH_HIDE_INPUTS": str(settings.langsmith_hide_inputs).lower(),
        "LANGSMITH_HIDE_OUTPUTS": str(settings.langsmith_hide_outputs).lower(),
    }
    if settings.langsmith_api_key:
        values["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    if settings.langsmith_workspace_id:
        values["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id
    os.environ.update(values)


async def build_runtime(settings: Settings) -> tuple[GroundedGraph, EventLedger, aiosqlite.Connection]:
    source = ensure_database(settings.database_path, settings.database_url, settings.database_sha256)
    database = ReadOnlyDatabase(source, row_limit=settings.sql_row_limit, timeout_ms=settings.sql_timeout_ms)
    retriever = HybridRetriever(database, settings.semantic_index_path, settings.local_embedding_model)
    await asyncio.to_thread(retriever.prepare)
    settings.checkpoint_database_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_connection = await aiosqlite.connect(settings.checkpoint_database_path)
    checkpointer = AsyncSqliteSaver(checkpoint_connection)
    await checkpointer.setup()
    graph = GroundedGraph(settings, retriever, checkpointer=checkpointer)
    return graph, EventLedger(settings.event_ledger_database_path), checkpoint_connection


async def run() -> None:
    settings = Settings()
    configure_langsmith_environment(settings)
    configure_logging(settings.log_level)
    graph, ledger, checkpoint_connection = await build_runtime(settings)
    handler = None
    bot = None
    try:
        _, bot, handler = await build_socket_app(settings, graph, ledger)
        structlog.get_logger().info("socket_mode_starting")
        await handler.start_async()
    except asyncio.CancelledError:
        pass
    finally:
        if handler is not None:
            await handler.close_async()
        if bot is not None:
            await bot.queue.close()
        await checkpoint_connection.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run the grounded Slack database bot in Socket Mode")
    parser.parse_args()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
