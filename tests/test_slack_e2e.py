from pathlib import Path
from typing import Any

import pytest

from slack_db_bot.config import Settings
from slack_db_bot.ledger import EventLedger
from slack_db_bot.models import AnswerBundle, Claim, ClaimLabel, Evidence
from slack_db_bot.slack_app import SlackBot


class FakeGraph:
    calls: list[tuple[str, list[dict[str, str]], str | None]]

    def __init__(self) -> None:
        self.calls = []

    async def answer(self, question: str, history: list[dict[str, str]], *, thread_id: str | None = None):
        self.calls.append((question, history, thread_id))
        evidence = [Evidence(evidence_id="art_1", source_type="test", title="Fixture", excerpt="BlueHarbor answer")]
        answer = AnswerBundle(claims=[Claim(text="BlueHarbor answer", label=ClaimLabel.FACT, evidence_ids=["art_1"])])
        return (
            answer,
            evidence,
            {"tool_calls": 1, "model_calls": 1, "rewrites": 0, "input_tokens": 10, "output_tokens": 5},
        )


class FakeSlackClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def reactions_add(self, **kwargs: Any) -> dict:
        self.calls.append(("reactions_add", kwargs))
        return {"ok": True}

    async def reactions_remove(self, **kwargs: Any) -> dict:
        self.calls.append(("reactions_remove", kwargs))
        return {"ok": True}

    async def chat_postMessage(self, **kwargs: Any) -> dict:
        self.calls.append(("chat_postMessage", kwargs))
        return {"ok": True, "ts": "2.0"}

    async def chat_update(self, **kwargs: Any) -> dict:
        self.calls.append(("chat_update", kwargs))
        return {"ok": True, "ts": kwargs["ts"]}

    async def files_upload_v2(self, **kwargs: Any) -> dict:
        self.calls.append(("files_upload_v2", kwargs))
        return {"ok": True}


class FailingGraph:
    async def answer(self, question: str, history: list[dict[str, str]], *, thread_id: str | None = None):
        raise TimeoutError("provider timed out")


class LongAnswerGraph:
    async def answer(self, question: str, history: list[dict[str, str]], *, thread_id: str | None = None):
        evidence = [Evidence(evidence_id="art_1", source_type="test", title="Fixture", excerpt="detail " * 400)]
        answer = AnswerBundle(
            claims=[Claim(text="detail " * 200, label=ClaimLabel.FACT, evidence_ids=["art_1"]) for _ in range(3)]
        )
        return answer, evidence, {"tool_calls": 1, "model_calls": 1, "rewrites": 0}


class FailingUploadClient(FakeSlackClient):
    async def files_upload_v2(self, **kwargs: Any) -> dict:
        self.calls.append(("files_upload_v2", kwargs))
        raise RuntimeError("upload failed")


def settings() -> Settings:
    return Settings(
        _env_file=None,
        slack_allowed_workspace_id="T1",
        slack_allowed_channel_ids="C1",
        database_path=Path(".data/synthetic_startup.sqlite"),
    )


@pytest.mark.asyncio
async def test_mention_then_unmentioned_followup_stays_in_thread_and_orders_history(tmp_path: Path) -> None:
    graph, client = FakeGraph(), FakeSlackClient()
    bot = SlackBot(settings(), graph, EventLedger(tmp_path / "events.sqlite"), client)  # type: ignore[arg-type]
    bot.bot_user_id = "B1"
    assert await bot.ingest(
        {"event_id": "Ev1", "team_id": "T1"},
        {"channel": "C1", "user": "U1", "ts": "1.0", "text": "<@B1> first question"},
        mentioned=True,
    )
    await bot.queue.join()
    assert await bot.ingest(
        {"event_id": "Ev2", "team_id": "T1"},
        {"channel": "C1", "user": "U1", "ts": "1.1", "thread_ts": "1.0", "text": "what about their milestone?"},
        mentioned=False,
    )
    await bot.queue.join()
    assert graph.calls[1][1][0]["text"] == "first question"
    posts = [payload for name, payload in client.calls if name == "chat_postMessage"]
    assert [post["thread_ts"] for post in posts] == ["1.0", "1.0"]
    assert all(not post["reply_broadcast"] for post in posts)


@pytest.mark.asyncio
async def test_unauthorized_unengaged_and_duplicate_events_are_ignored(tmp_path: Path) -> None:
    graph, client = FakeGraph(), FakeSlackClient()
    bot = SlackBot(settings(), graph, EventLedger(tmp_path / "events.sqlite"), client)  # type: ignore[arg-type]
    event = {"channel": "C2", "user": "U1", "ts": "1.0", "text": "<@B1> secret"}
    assert not await bot.ingest({"event_id": "Ev0", "team_id": "T1"}, event, mentioned=True)
    followup = {"channel": "C1", "user": "U1", "ts": "1.1", "thread_ts": "1.0", "text": "hello"}
    assert not await bot.ingest({"event_id": "Ev1", "team_id": "T1"}, followup, mentioned=False)
    mention = {"channel": "C1", "user": "U1", "ts": "1.0", "text": "<@B1> hello"}
    assert await bot.ingest({"event_id": "Ev2", "team_id": "T1"}, mention, mentioned=True)
    assert not await bot.ingest({"event_id": "Ev2", "team_id": "T1"}, mention, mentioned=True)
    await bot.queue.join()
    assert len(graph.calls) == 1


@pytest.mark.asyncio
async def test_explicit_broadcast_is_the_only_broadcast(tmp_path: Path) -> None:
    graph, client = FakeGraph(), FakeSlackClient()
    bot = SlackBot(settings(), graph, EventLedger(tmp_path / "events.sqlite"), client)  # type: ignore[arg-type]
    await bot.ingest(
        {"event_id": "Ev1", "team_id": "T1"},
        {"channel": "C1", "user": "U1", "ts": "1.0", "text": "<@B1> answer and share this in the channel"},
        mentioned=True,
    )
    await bot.queue.join()
    post = next(payload for name, payload in client.calls if name == "chat_postMessage")
    assert post["reply_broadcast"] is True


@pytest.mark.asyncio
async def test_dm_and_negated_broadcast_are_not_expanded(tmp_path: Path) -> None:
    graph, client = FakeGraph(), FakeSlackClient()
    bot = SlackBot(settings(), graph, EventLedger(tmp_path / "events.sqlite"), client)  # type: ignore[arg-type]
    event = {"channel": "C1", "channel_type": "im", "user": "U1", "ts": "1.0", "text": "<@B1> secret"}
    assert not await bot.ingest({"event_id": "Ev0", "team_id": "T1"}, event, mentioned=True)
    await bot.ingest(
        {"event_id": "Ev1", "team_id": "T1"},
        {"channel": "C1", "user": "U1", "ts": "1.1", "text": "<@B1> don't share this in the channel"},
        mentioned=True,
    )
    await bot.queue.join()
    post = next(payload for name, payload in client.calls if name == "chat_postMessage")
    assert post["reply_broadcast"] is False


@pytest.mark.asyncio
async def test_dependency_failure_posts_clear_thread_error(tmp_path: Path) -> None:
    client = FakeSlackClient()
    bot = SlackBot(settings(), FailingGraph(), EventLedger(tmp_path / "events.sqlite"), client)  # type: ignore[arg-type]
    await bot.ingest(
        {"event_id": "Ev1", "team_id": "T1"},
        {"channel": "C1", "user": "U1", "ts": "1.0", "text": "<@B1> question"},
        mentioned=True,
    )
    await bot.queue.join()
    post = next(payload for name, payload in client.calls if name == "chat_postMessage")
    assert post["thread_ts"] == "1.0"
    assert "exceeded its time budget" in post["text"]
    assert any(name == "reactions_remove" for name, _ in client.calls)


@pytest.mark.asyncio
async def test_long_answer_posts_summary_and_attachment(tmp_path: Path) -> None:
    client = FakeSlackClient()
    bot = SlackBot(settings(), LongAnswerGraph(), EventLedger(tmp_path / "events.sqlite"), client)  # type: ignore[arg-type]
    await bot.ingest(
        {"event_id": "Ev1", "team_id": "T1"},
        {"channel": "C1", "user": "U1", "ts": "1.0", "text": "<@B1> long answer"},
        mentioned=True,
    )
    await bot.queue.join()
    post = next(payload for name, payload in client.calls if name == "chat_postMessage")
    upload = next(payload for name, payload in client.calls if name == "files_upload_v2")
    assert "Full answer attached" in post["text"]
    assert upload["thread_ts"] == "1.0"
    assert len(upload["content"]) > 3_500


@pytest.mark.asyncio
async def test_attachment_failure_updates_the_existing_message_without_duplicate_post(tmp_path: Path) -> None:
    client = FailingUploadClient()
    ledger = EventLedger(tmp_path / "events.sqlite")
    bot = SlackBot(settings(), LongAnswerGraph(), ledger, client)  # type: ignore[arg-type]
    await bot.ingest(
        {"event_id": "Ev1", "team_id": "T1"},
        {"channel": "C1", "user": "U1", "ts": "1.0", "text": "<@B1> long answer"},
        mentioned=True,
    )
    await bot.queue.join()
    assert sum(name == "chat_postMessage" for name, _ in client.calls) == 1
    assert sum(name == "chat_update" for name, _ in client.calls) == 1
    update = next(payload for name, payload in client.calls if name == "chat_update")
    assert "could not be confirmed" in update["text"]
    delivery = ledger.delivery("Ev1")
    assert delivery is not None
    assert delivery["message_ts"] == "2.0"
