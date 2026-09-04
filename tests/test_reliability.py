from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from slack_sdk.errors import SlackApiError

from slack_db_bot.answering import GroundedGraph
from slack_db_bot.config import Settings
from slack_db_bot.models import AnswerBundle, Claim, ClaimLabel, Evidence
from slack_db_bot.slack_app import SlackCalls, SlackTurn, ThreadQueue


def turn(event_id: str, root_ts: str) -> SlackTurn:
    return SlackTurn(event_id, "T1", "C1", "U1", event_id, root_ts, event_id, False)


def test_all_default_openai_roles_use_gpt54_mini() -> None:
    settings = Settings(_env_file=None)
    assert {
        settings.openai_router_model,
        settings.openai_synthesis_model,
        settings.openai_eval_model,
    } == {"gpt-5.4-mini"}


@pytest.mark.asyncio
async def test_thread_queue_orders_each_thread_and_runs_independent_threads_concurrently() -> None:
    active = 0
    peak = 0
    events: list[str] = []

    async def process(item: SlackTurn) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        events.append("start:" + item.event_id)
        await asyncio.sleep(0.02)
        events.append("end:" + item.event_id)
        active -= 1

    queue = ThreadQueue(process, capacity=4, concurrency=2)
    queue.submit(turn("a1", "thread-a"))
    queue.submit(turn("a2", "thread-a"))
    queue.submit(turn("b1", "thread-b"))
    await queue.join()
    assert events.index("end:a1") < events.index("start:a2")
    assert peak == 2


@pytest.mark.asyncio
async def test_thread_queue_rejects_work_over_capacity() -> None:
    release = asyncio.Event()

    async def process(_: SlackTurn) -> None:
        await release.wait()

    queue = ThreadQueue(process, capacity=1, concurrency=1)
    queue.submit(turn("a1", "thread-a"))
    with pytest.raises(RuntimeError, match="queue is full"):
        queue.submit(turn("b1", "thread-b"))
    release.set()
    await queue.join()


class RateLimitedClient:
    calls = 0

    async def chat_postMessage(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            response = type("Response", (), {"status_code": 429, "headers": {"Retry-After": "0"}})()
            raise SlackApiError("rate limited", response)  # type: ignore[arg-type]
        return {"ok": True, "ts": "2.0"}


@pytest.mark.asyncio
async def test_slack_calls_retries_one_short_rate_limit_and_counts_it() -> None:
    client = RateLimitedClient()
    calls = SlackCalls(client, limit=2)  # type: ignore[arg-type]
    result = await calls.call("chat_postMessage", text="hello")
    assert result["ts"] == "2.0"
    assert client.calls == 2
    assert calls.used == 2


class StaticRetriever:
    def search(self, _: str, *, limit: int, diversify: bool = True) -> list[Evidence]:
        return [
            Evidence(
                evidence_id="art_test",
                source_type="internal_document",
                title="Verified plan",
                customer_name="Example Co",
                excerpt="Example Co approved the patch.",
            )
        ][:limit]


class StructuredModel:
    def with_structured_output(self, *_: Any, **__: Any) -> StructuredModel:
        return self

    async def ainvoke(self, *_: Any, **__: Any) -> AnswerBundle:
        return AnswerBundle(
            claims=[Claim(text="Example Co approved the patch.", label=ClaimLabel.FACT, evidence_ids=["art_test"])]
        )


@pytest.mark.asyncio
async def test_real_graph_routes_retrieves_synthesizes_and_validates_with_mocked_model() -> None:
    settings = Settings(_env_file=None, database_path=Path("unused"))
    model = StructuredModel()
    graph = GroundedGraph(
        settings,
        cast(Any, StaticRetriever()),
        direct_model=cast(Any, model),
        synthesis_model=cast(Any, model),
    )
    answer, evidence, metrics = await graph.answer("What did Example Co approve?")
    assert answer.claims[0].text == "Example Co approved the patch."
    assert evidence[0].evidence_id == "art_test"
    assert metrics["tool_calls"] == 1
