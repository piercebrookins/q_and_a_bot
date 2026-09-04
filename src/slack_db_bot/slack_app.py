from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from .answering import GroundedGraph, render_answer
from .config import Settings
from .ledger import EventLedger

log = structlog.get_logger()
BROADCAST = re.compile(
    r"(?:^|[.!?]\s*|\band\s+)(?:please\s+)?(?:broadcast|post|share)\b.{0,24}\b(?:channel|everyone)\b",
    re.IGNORECASE,
)


class SlackCalls:
    def __init__(self, client: AsyncWebClient, limit: int):
        self.client, self.limit, self.used = client, limit, 0

    async def call(self, method: str, *, cost: int = 1, **kwargs: Any) -> Any:
        for attempt in range(2):
            if self.used + cost > self.limit:
                raise RuntimeError("Slack operation budget exhausted")
            self.used += cost
            try:
                async with asyncio.timeout(15):
                    return await getattr(self.client, method)(**kwargs)
            except SlackApiError as exc:
                if exc.response.status_code != 429 or attempt or cost != 1:
                    raise
                delay = float(exc.response.headers.get("Retry-After", "1"))
                if delay > 3:
                    raise
                await asyncio.sleep(max(0, delay))


@dataclass(frozen=True)
class SlackTurn:
    event_id: str
    team_id: str
    channel_id: str
    user_id: str
    message_ts: str
    root_ts: str
    text: str
    broadcast: bool

    @property
    def thread_key(self) -> str:
        return EventLedger.thread_key(self.team_id, self.channel_id, self.root_ts)


class ThreadQueue:
    def __init__(self, process: Callable[[SlackTurn], Awaitable[None]], *, capacity: int = 32, concurrency: int = 4):
        self.process = process
        self.capacity, self.pending = capacity, 0
        self.semaphore = asyncio.Semaphore(concurrency)
        self.queues: dict[str, asyncio.Queue[SlackTurn]] = {}
        self.workers: dict[str, asyncio.Task[None]] = {}

    def submit(self, turn: SlackTurn) -> None:
        if self.pending >= self.capacity:
            raise RuntimeError("Thread queue is full")
        self.pending += 1
        queue = self.queues.setdefault(turn.thread_key, asyncio.Queue())
        queue.put_nowait(turn)
        worker = self.workers.get(turn.thread_key)
        if worker is None or worker.done():
            self.workers[turn.thread_key] = asyncio.create_task(self._run(turn.thread_key))

    async def _run(self, key: str) -> None:
        queue = self.queues[key]
        while not queue.empty():
            turn = await queue.get()
            try:
                async with self.semaphore:
                    await self.process(turn)
            except Exception as exc:  # noqa: BLE001 -- isolate one worker failure from queued turns
                log.error("queue_worker_failed", error=type(exc).__name__)
            finally:
                self.pending -= 1
                queue.task_done()
        self.queues.pop(key, None)
        self.workers.pop(key, None)

    async def join(self) -> None:
        await asyncio.gather(*(queue.join() for queue in list(self.queues.values())))

    async def close(self) -> None:
        workers = list(self.workers.values())
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        self.queues.clear()
        self.workers.clear()
        self.pending = 0


class SlackBot:
    def __init__(self, settings: Settings, graph: GroundedGraph, ledger: EventLedger, client: AsyncWebClient):
        self.settings = settings
        self.graph = graph
        self.ledger = ledger
        self.client = client
        self.queue = ThreadQueue(
            self.process, capacity=settings.max_pending_turns, concurrency=settings.max_concurrent_turns
        )
        self.last_prune = time.monotonic()
        self.bot_user_id = ""

    def authorized(self, team_id: str, channel_id: str) -> bool:
        return team_id == self.settings.slack_allowed_workspace_id and channel_id in self.settings.allowed_channels

    async def ingest(self, body: dict[str, Any], event: dict[str, Any], *, mentioned: bool) -> bool:
        event_id = str(body.get("event_id", ""))
        team_id = str(body.get("team_id") or event.get("team", ""))
        channel_id = str(event.get("channel", ""))
        if not event_id or not self.authorized(team_id, channel_id):
            log.warning("event_rejected", event_id=event_id or None, reason="unauthorized_context")
            return False
        if event.get("bot_id") or event.get("subtype"):
            return False
        if not event.get("user") or event.get("channel_type") == "im":
            return False
        text = str(event.get("text", ""))
        if len(text) > 6000 or self.queue.pending >= self.queue.capacity:
            log.warning("event_rejected", event_id=event_id, reason="input_or_queue_budget")
            return False
        if time.monotonic() - self.last_prune > 3600:
            self.ledger.prune()
            self.last_prune = time.monotonic()
        if not mentioned and self.bot_user_id and f"<@{self.bot_user_id}>" in text:
            return False
        message_ts = str(event.get("ts", ""))
        root_ts = str(event.get("thread_ts") or message_ts)
        if not message_ts or not root_ts:
            return False
        key = EventLedger.thread_key(team_id, channel_id, root_ts)
        if mentioned:
            self.ledger.engage(key)
        elif not event.get("thread_ts") or not self.ledger.is_engaged(key):
            return False
        if not self.ledger.claim_event(event_id, key):
            log.info("duplicate_event", event_id=event_id, thread_key=key[:12])
            return False
        turn = SlackTurn(
            event_id=event_id,
            team_id=team_id,
            channel_id=channel_id,
            user_id=str(event.get("user", "")),
            message_ts=message_ts,
            root_ts=root_ts,
            text=re.sub(r"<@[A-Z0-9]+>", "", text).strip(),
            broadcast=bool(BROADCAST.search(text)),
        )
        self.queue.submit(turn)
        log.info("event_queued", event_id=event_id, thread_key=key[:12], mentioned=mentioned)
        return True

    async def process(self, turn: SlackTurn) -> None:
        started = time.monotonic()
        calls = SlackCalls(self.client, self.settings.max_slack_calls)
        delivering = False
        try:
            try:
                await calls.call("reactions_add", channel=turn.channel_id, timestamp=turn.message_ts, name="eyes")
            except Exception as exc:  # noqa: BLE001 -- progress must not block the answer
                log.warning("progress_reaction_failed", event_id=turn.event_id, error=type(exc).__name__)
            history = self.ledger.history(turn.thread_key)
            async with asyncio.timeout(self.settings.turn_timeout_seconds):
                answer, evidence, metrics = await self.graph.answer(turn.text, history, thread_id=turn.thread_key)
            text = render_answer(answer, evidence)
            delivering = True
            await self._deliver(turn, text, calls)
            self.ledger.append_turns(turn.thread_key, turn.text, text)
            latency_ms = round((time.monotonic() - started) * 1000)
            self.ledger.record_metrics(turn.event_id, metrics, latency_ms)
            log.info(
                "turn_complete",
                event_id=turn.event_id,
                thread_key=turn.thread_key[:12],
                latency_ms=latency_ms,
                **metrics,
            )
        except Exception as exc:  # noqa: BLE001 -- final turn boundary maps all dependency errors
            log.error("turn_failed", event_id=turn.event_id, thread_key=turn.thread_key[:12], error=type(exc).__name__)
            if delivering:
                if not self.ledger.delivery(turn.event_id):
                    self.ledger.mark_uncertain(turn.event_id)
            else:
                self.ledger.mark_failed(turn.event_id, type(exc).__name__)
                message = (
                    "The database lookup exceeded its time budget. Please ask a narrower question."
                    if isinstance(exc, TimeoutError)
                    else "I couldn't complete that database lookup because a dependency failed. "
                    "Please send a new message to retry."
                )
                await self._deliver_error(turn, message, calls)
        finally:
            try:
                await calls.call("reactions_remove", channel=turn.channel_id, timestamp=turn.message_ts, name="eyes")
            except Exception as exc:  # noqa: BLE001 -- cleanup is best effort
                log.debug("progress_reaction_remove_failed", event_id=turn.event_id, error=type(exc).__name__)

    async def _deliver(self, turn: SlackTurn, text: str, calls: SlackCalls) -> None:
        existing = self.ledger.delivery(turn.event_id)
        if existing:
            if existing["channel_id"] != turn.channel_id:
                raise ValueError("Delivery channel does not match authenticated context")
            result = await calls.call(
                "chat_update", channel=existing["channel_id"], ts=existing["message_ts"], text=text[:3900]
            )
            message_ts = str(result["ts"])
        else:
            visible = text if len(text) <= 3500 else text[:3200] + "\n\n_Full answer attached in this thread._"
            result = await calls.call(
                "chat_postMessage",
                channel=turn.channel_id,
                thread_ts=turn.root_ts,
                text=visible,
                reply_broadcast=turn.broadcast,
                client_msg_id=str(uuid.uuid5(uuid.NAMESPACE_URL, turn.event_id)),
                unfurl_links=False,
                unfurl_media=False,
            )
            message_ts = str(result["ts"])
            self.ledger.mark_delivered(turn.event_id, turn.channel_id, message_ts, visible)
            if len(text) > 3500:
                try:
                    await calls.call(
                        "files_upload_v2",
                        cost=3,
                        channel=turn.channel_id,
                        thread_ts=turn.root_ts,
                        content=text,
                        filename=f"answer-{turn.event_id}.md",
                        title="Full grounded answer",
                    )
                except Exception as exc:  # noqa: BLE001 -- preserve already delivered answer on attachment failure
                    log.error("attachment_delivery_failed", event_id=turn.event_id, error=type(exc).__name__)
                    visible = visible.replace(
                        "Full answer attached in this thread.",
                        "Attachment delivery could not be confirmed; ask a narrower follow-up "
                        "for the remaining details.",
                    )
                    await calls.call("chat_update", channel=turn.channel_id, ts=message_ts, text=visible)
                    self.ledger.mark_delivered(turn.event_id, turn.channel_id, message_ts, visible)
                    return
        self.ledger.mark_delivered(turn.event_id, turn.channel_id, message_ts, text)

    async def _deliver_error(self, turn: SlackTurn, message: str, calls: SlackCalls) -> None:
        try:
            await self._deliver(turn, message, calls)
        except Exception as exc:  # noqa: BLE001 -- final failure boundary
            log.error("error_delivery_failed", event_id=turn.event_id, error=type(exc).__name__)


async def build_socket_app(
    settings: Settings, graph: GroundedGraph, ledger: EventLedger
) -> tuple[AsyncApp, SlackBot, AsyncSocketModeHandler]:
    settings.require_live()
    if settings.slack_bot_token is None or settings.slack_app_token is None:
        raise ValueError("Slack tokens are required")
    app = AsyncApp(
        client=AsyncWebClient(token=settings.slack_bot_token.get_secret_value(), timeout=15, retry_handlers=[])
    )
    bot = SlackBot(settings, graph, ledger, app.client)
    auth = await app.client.auth_test()
    if str(auth["team_id"]) != settings.slack_allowed_workspace_id:
        raise ValueError("Slack token belongs to a different workspace")
    bot.bot_user_id = str(auth["user_id"])

    @app.event("app_mention")
    async def on_mention(body: dict, event: dict, ack: Callable[..., Awaitable[Any]]) -> None:
        await ack()
        await bot.ingest(body, event, mentioned=True)

    @app.event("message")
    async def on_message(body: dict, event: dict, ack: Callable[..., Awaitable[Any]]) -> None:
        await ack()
        await bot.ingest(body, event, mentioned=False)

    handler = AsyncSocketModeHandler(app, settings.slack_app_token.get_secret_value())
    return app, bot, handler
