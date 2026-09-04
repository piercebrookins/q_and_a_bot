from collections.abc import Sequence
from pathlib import Path

import numpy as np

from slack_db_bot.database import ReadOnlyDatabase
from slack_db_bot.retrieval import HybridRetriever


class DeterministicEmbedder:
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[len(text) % 17 + 1, len(text) % 29 + 1] for text in texts], dtype=np.float32)


def retriever(tmp_path: Path, database_path: Path) -> HybridRetriever:
    return HybridRetriever(
        ReadOnlyDatabase(database_path),
        tmp_path / "index",
        "deterministic-test",
        embedder=DeterministicEmbedder(),
    )


def test_hybrid_retrieval_finds_exact_customer_and_command(tmp_path: Path, database_path: Path) -> None:
    evidence = retriever(tmp_path, database_path).search(
        "For Verdant Bay, what is the approved patch window and orchestrator rollback command?", limit=8
    )
    ids = {item.evidence_id for item in evidence}
    assert any(item.startswith("structured:cus_b430f59e0caf:") for item in ids)
    assert "art_fff67d92fe41" in ids
    playbook = next(item for item in evidence if item.evidence_id == "art_fff67d92fe41")
    assert "orchestrator rollback --target ruleset=<prior_sha>" in playbook.excerpt


def test_hyphenated_text_is_safely_quoted_for_fts(tmp_path: Path, database_path: Path) -> None:
    evidence = retriever(tmp_path, database_path).search("7-10 business-day proof-of-fix", limit=8)
    assert any(item.customer_name == "BlueHarbor Logistics" for item in evidence)


def test_distinctive_entity_token_does_not_match_substring(tmp_path: Path, database_path: Path) -> None:
    evidence = retriever(tmp_path, database_path).search("MapleHarvest Quebec pilot", limit=8)
    structured = [item.customer_name for item in evidence if item.source_type == "structured account record"]
    assert structured == ["MapleHarvest Grocers"]


def test_diversified_search_returns_one_artifact_per_customer(tmp_path: Path, database_path: Path) -> None:
    evidence = retriever(tmp_path, database_path).search(
        "North America West Event Nexus taxonomy search semantics duplicate actions", limit=20, diversify=True
    )
    artifact_customers = [item.customer_name for item in evidence if not item.evidence_id.startswith("structured:")]
    assert len(artifact_customers) == len(set(artifact_customers))
