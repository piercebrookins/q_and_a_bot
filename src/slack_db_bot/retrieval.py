from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding

from .database import ReadOnlyDatabase, terms
from .models import Evidence

STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "among",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "bot",
        "by",
        "do",
        "does",
        "for",
        "from",
        "give",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "our",
        "really",
        "so",
        "that",
        "the",
        "their",
        "them",
        "this",
        "to",
        "versus",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
    }
)

QUERY_EXPANSIONS = {
    "cheaper": "low-cost budget buy time",
    "tactical": "stopgap tactical layer",
    "defect": "switch churn renewal retention risk",
    "milestone": "proof acceptance promised deadline",
    "taxonomy": "search relevance semantics ranking",
    "duplicate-action": "duplicate incidents repeated playbook executions idempotency",
    "approval-bypass": "approval routing rules precedence country default stale cache schema alias",
}


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)


def _normalized(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _excerpt(text: str, query: str, max_chars: int = 3_600) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    query_terms = [x for x in terms(query) if x not in STOPWORDS and len(x) > 2]
    positions = [compact.lower().find(term) for term in query_terms]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - max_chars // 4)
    end = min(len(compact), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    return ("…" if start else "") + compact[start:end] + ("…" if end < len(compact) else "")


class HybridRetriever:
    def __init__(
        self,
        database: ReadOnlyDatabase,
        index_path: Path,
        model_name: str,
        *,
        embedder: Embedder | None = None,
    ):
        self.database = database
        self.index_path = index_path
        self.model_name = model_name
        self._embedder = embedder
        self._artifacts: list[dict] | None = None
        self._vectors: np.ndarray | None = None

    @property
    def artifacts(self) -> list[dict]:
        if self._artifacts is None:
            self._artifacts = self.database.all_artifacts()
        return self._artifacts

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = FastEmbedder(self.model_name)
        return self._embedder

    def prepare(self) -> None:
        fingerprint = hashlib.sha256(
            (self.model_name + "\n" + "\n".join(x["artifact_id"] + x["content_text"] for x in self.artifacts)).encode()
        ).hexdigest()
        vectors_file = self.index_path / "vectors.npz"
        self.index_path.mkdir(parents=True, exist_ok=True)
        if vectors_file.exists():
            loaded = np.load(vectors_file, allow_pickle=False)
            if str(loaded["fingerprint"].item()) == fingerprint:
                self._vectors = loaded["vectors"]
                return
        texts = [f"{row['title']}\n{row['summary']}\n{row['content_text']}" for row in self.artifacts]
        self._vectors = _normalized(self.embedder.embed(texts))
        np.savez_compressed(vectors_file, vectors=self._vectors, fingerprint=np.asarray(fingerprint))

    def _semantic(self, query: str, limit: int) -> list[tuple[str, float]]:
        if self._vectors is None:
            self.prepare()
        if self._vectors is None:
            raise RuntimeError("Semantic index was not prepared")
        query_vector = _normalized(self.embedder.embed([query]))[0]
        scores = self._vectors @ query_vector
        indices = np.argsort(scores)[::-1][:limit]
        return [(self.artifacts[int(index)]["artifact_id"], float(scores[int(index)])) for index in indices]

    def _lexical(self, query: str, limit: int) -> list[tuple[str, float]]:
        tokens = []
        for token in terms(query):
            if token not in STOPWORDS and len(token) > 2 and token not in tokens:
                tokens.append(token)
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens[:16])
        rows = self.database.query(
            """SELECT artifact_id, bm25(artifacts_fts) AS rank
               FROM artifacts_fts WHERE artifacts_fts MATCH ? ORDER BY rank LIMIT 60""",
            (expression,),
        )
        return [(row["artifact_id"], -float(row["rank"])) for row in rows[:limit]]

    def _named_customers(self, query: str) -> list[dict]:
        customers = self.database.query("SELECT customer_id, scenario_id, name FROM customers LIMIT 100")
        lowered = query.lower()
        query_terms = set(terms(query))
        matched = []
        for row in customers:
            name = row["name"].lower()
            distinctive = [token for token in terms(name) if len(token) >= 6]
            if name in lowered or any(token in query_terms for token in distinctive):
                matched.append(row)
        return matched

    def _structured(self, named: Iterable[dict]) -> list[Evidence]:
        result: list[Evidence] = []
        for customer in named:
            rows = self.database.query(
                """SELECT c.customer_id, c.name, c.industry, c.region, c.country, c.crm_stage,
                          c.account_health, c.tech_stack_summary, c.notes, s.scenario_id, s.trigger_event,
                          s.pain_point, s.scenario_summary, s.status AS scenario_status,
                          pc.name AS primary_competitor_name,
                          i.implementation_id, i.status AS implementation_status, i.kickoff_date,
                          i.go_live_date, i.scope_summary, i.success_metrics_json,
                          i.risks_json
                   FROM customers AS c LEFT JOIN scenarios AS s ON s.scenario_id = c.scenario_id
                   LEFT JOIN competitors AS pc ON pc.competitor_id = s.primary_competitor_id
                   LEFT JOIN implementations AS i ON i.customer_id = c.customer_id
                   WHERE c.customer_id = ? ORDER BY i.implementation_id LIMIT 100""",
                (customer["customer_id"],),
            )
            for row in rows:
                text = "; ".join(f"{key}={value}" for key, value in row.items() if value not in (None, ""))
                result.append(
                    Evidence(
                        evidence_id=f"structured:{row['customer_id']}:{row['implementation_id'] or 'account'}",
                        source_type="structured account record",
                        title=f"Account record: {row['name']}",
                        customer_name=row["name"],
                        region=row["region"],
                        country=row["country"],
                        excerpt=text,
                        score=2.0,
                    )
                )
        return result

    def search(self, query: str, *, limit: int = 16, diversify: bool = False) -> list[Evidence]:
        lowered = query.lower()
        additions = [value for marker, value in QUERY_EXPANSIONS.items() if marker in lowered]
        expanded_query = query + (" " + " ".join(additions) if additions else "")
        candidate_limit = min(max(limit * 3, 24), 60)
        lexical = self._lexical(expanded_query, candidate_limit)
        semantic = self._semantic(expanded_query, candidate_limit)
        ranks: dict[str, float] = {}
        raw_scores: dict[str, float] = {}
        for source in (lexical, semantic):
            for rank, (artifact_id, score) in enumerate(source, 1):
                ranks[artifact_id] = ranks.get(artifact_id, 0.0) + 1.0 / (60 + rank)
                raw_scores[artifact_id] = max(raw_scores.get(artifact_id, -1.0), score)

        named = self._named_customers(query)
        names = {row["name"] for row in named}
        artifacts_by_id = {row["artifact_id"]: row for row in self.artifacts}
        expanded_terms = {token for token in terms(expanded_query) if token not in STOPWORDS and len(token) > 2}
        query_dates = set(re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", query))
        for artifact_id in list(ranks):
            row = artifacts_by_id[artifact_id]
            if row.get("customer_name") in names:
                ranks[artifact_id] += 0.08
            searchable = f"{row['title']} {row['summary']} {row['content_text']}".lower()
            if expanded_terms:
                coverage = len(expanded_terms & set(terms(searchable))) / len(expanded_terms)
                ranks[artifact_id] += 0.03 * coverage
            if query_dates and any(date in searchable for date in query_dates):
                ranks[artifact_id] += 0.05
            if "approved" in searchable or "final" in searchable:
                ranks[artifact_id] += 0.003

        ordered = sorted(ranks, key=lambda item: (ranks[item], raw_scores[item]), reverse=True)
        evidence = self._structured(named)
        customer_counts: dict[str, int] = {}
        for artifact_id in ordered:
            row = artifacts_by_id[artifact_id]
            customer = row.get("customer_name") or f"artifact:{artifact_id}"
            if diversify and customer_counts.get(customer, 0) >= 1:
                continue
            evidence.append(
                Evidence(
                    evidence_id=artifact_id,
                    source_type=row["artifact_type"],
                    title=row["title"],
                    created_at=row["created_at"],
                    customer_name=row.get("customer_name"),
                    region=row.get("region"),
                    country=row.get("country"),
                    excerpt=_excerpt(
                        f"{row['summary']}\n{row['content_text']}", query, max_chars=1_800 if diversify else 3_600
                    ),
                    score=ranks[artifact_id],
                )
            )
            customer_counts[customer] = customer_counts.get(customer, 0) + 1
            if len(evidence) >= limit:
                break
        return evidence[:limit]
