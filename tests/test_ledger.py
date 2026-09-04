from pathlib import Path

from slack_db_bot.ledger import EventLedger


def test_deduplication_delivery_and_restart_state(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite"
    key = EventLedger.thread_key("T1", "C1", "1.0")
    first = EventLedger(path)
    assert first.claim_event("Ev1", key)
    assert not first.claim_event("Ev1", key)
    first.engage(key)
    first.mark_delivered("Ev1", "C1", "2.0", "answer")
    first.append_turns(key, "question", "answer")

    restarted = EventLedger(path)
    assert restarted.is_engaged(key)
    assert not restarted.claim_event("Ev1", key)
    delivery = restarted.delivery("Ev1")
    assert delivery is not None
    assert delivery["message_ts"] == "2.0"
    assert restarted.history(key) == [{"role": "user", "text": "question"}, {"role": "assistant", "text": "answer"}]


def test_failed_event_can_be_retried(tmp_path: Path) -> None:
    ledger = EventLedger(tmp_path / "events.sqlite")
    assert ledger.claim_event("Ev1", "key")
    ledger.mark_failed("Ev1", "dependency")
    assert ledger.claim_event("Ev1", "key")


def test_processing_event_can_be_reclaimed_after_crash_timeout(tmp_path: Path) -> None:
    ledger = EventLedger(tmp_path / "events.sqlite")
    assert ledger.claim_event("Ev1", "key")
    with ledger.connect() as connection:
        connection.execute("UPDATE events SET updated_at=datetime('now', '-11 minutes') WHERE event_id='Ev1'")
        connection.commit()
    assert ledger.claim_event("Ev1", "key")
    assert not ledger.claim_event("Ev1", "key")


def test_long_thread_history_is_compacted_and_bounded(tmp_path: Path) -> None:
    ledger = EventLedger(tmp_path / "events.sqlite")
    for index in range(8):
        ledger.append_turns("key", f"question {index}", f"answer {index}")
    history = ledger.history("key")
    assert history[0]["text"].startswith("Earlier thread context")
    assert "question 0" in history[0]["text"]
    assert len(history) == 13
    assert history[-1]["text"] == "answer 7"
