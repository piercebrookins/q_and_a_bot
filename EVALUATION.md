# Evaluation results

All reported figures below were measured on 2026-09-03. Every OpenAI role in the final configuration uses
`gpt-5.4-mini`: direct answering, investigation, synthesis, bounded repair, and semantic evaluation.

## Automated gates

- Ruff formatting and lint passed.
- Mypy found no issues in 17 source and test files.
- Pytest passed 59 tests with 70% total line coverage (retrieval 93%, ledger 96%, Slack adapter 78%).
- `uvx pip-audit --strict` found no known vulnerabilities.
- `uv lock --check` passed.
- Both the source distribution and wheel built successfully.
- The source SQLite SHA-256 is `5bd743daf068f55599e0b93f97f65973298c7123c9d67518f533bd0aa2925c2a`.

## Assignment acceptance cases

[The final local result](evaluation/results/latest.json) records every rendered answer, cited evidence, deterministic
concept grade, structured semantic verdict, latency, calls, and tokens. It completed at
`2026-09-03T20:05:24.662193Z`.

| Configuration | Passed | Concept recall | Mean graph latency | Input tokens | Output tokens | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.4 mini for every OpenAI role; `BAAI/bge-small-en-v1.5` embeddings | 7/7 | 100% | 7,163 ms | 144,607 | 5,850 | 18 |

Token totals include generation and semantic evaluation. The quality gate requires all expected concepts, a
non-abstaining answer, action and latency budgets, and a semantic verdict that is correct, complete, grounded,
correctly associated, contradiction-free, and free of unsupported claims. Exact dollar cost is not reported because
captured usage does not separate every billable cache and reasoning category.

The all-mini work exposed real failure modes before the final pass: paraphrased exact terms, cohort representatives
displaced by follow-up documents, irrelevant timeline clauses copied during deterministic repair, and a semantic judge
that sometimes treated supported details as unsupported. The release code now preserves named plan fields, prioritizes
one representative per requested cohort, renders compact source-backed answers for exhaustive classifications, and
treats the supplied acceptance concepts and gold evidence as the evaluation reference. The saved final result is the
complete rerun after those fixes.

## LangSmith

The final experiment,
[`slack-db-bot-gpt-5.4-mini-gpt-5.4-mini-0b4a928b`](https://smith.langchain.com/o/b966086f-707d-4439-bdef-9e441ce1ceee/datasets/29233e7f-4918-473a-a608-b397859e763a/compare?selectedSessions=83b07b2b-1010-40d6-8fa3-b41a7154bab7),
ran the real graph for all seven dataset examples and scored 7/7 quality gates. This was verified by querying its seven
root runs and confirming a `quality_gate` score of `1.0` on each. LangSmith inputs and outputs remain hidden by the
configured trace-privacy settings.

The immediately preceding all-mini experiment scored 6/7. Its failure showed that the superlative-risk answer cited
the selected customer but not the candidates used for comparison. The final implementation cites the comparison
evidence and preserves the conclusion as an inference. The 7/7 experiment above supersedes that run.

## Live Slack and OpenAI

[The live smoke artifact](evaluation/results/live-smoke.json) records the final all-mini test. A real Slack mention in
the allowed demo channel reached the bot through Socket Mode. The bot used one
retrieval, two GPT-5.4 mini calls including one grounded repair, and posted one threaded answer in 3,599 ms. Logged
usage was 24,287 input tokens and 254 output tokens.

The visible response contained the approved `2026-03-24 02:00-04:00` window, exact
`orchestrator rollback --target ruleset=<prior_sha>` command, rehydration and invalidation-hook behavior, and a readable
source title. No raw evidence ID appeared in the Slack reply. Socket Mode then shut down cleanly without a traceback.

## Limits

The application is a single local process. Its persistent ledger deduplicates delivered events and reclaims a stale
processing claim after ten minutes, but queued work is not durable across a crash. Workspace and channel authorization
match the assignment's synthetic-data scope; row-level and per-user entitlements are not implemented. Direct messages
and feedback UI are excluded by the clarified scope. The seven acceptance prompts are task-specific, while separate
tests cover prompt injection, malformed model claims, authorization, duplicate delivery, bounded memory, and dependency
failure. A successful model-based evaluation is evidence for this run and does not imply deterministic future outputs.
