# Grounded Slack database Q&A bot

This repository runs a local Slack Socket Mode bot over the supplied synthetic SQLite database. It answers explicit
mentions and unmentioned follow-ups in an already engaged thread, shows readable source titles backed by stable
internal evidence IDs, labels inference, exposes gaps and conflicts, and refuses unsupported questions.

The implementation uses an explicit LangGraph `StateGraph`. Every generation, investigation, synthesis, repair, and
evaluation model call uses GPT-5.4 mini. Direct questions use one hybrid retrieval pass. Comparisons, recurring
patterns, and superlatives use a bounded Deep Agents investigator before grounded synthesis. Retrieval combines
read-only structured SQLite, FTS5, and a local FastEmbed index with reciprocal rank fusion and deterministic entity,
date, and coverage rules. LangSmith records privacy-redacted execution traces and runs the seven-case evaluation suite.

## LangSmith tracing and evaluation

LangSmith is integrated into both the application and its evaluation workflow. With `LANGSMITH_TRACING=true`, graph,
model, and `search_database` spans are recorded under the configured project while
`LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true` keep Slack messages, database excerpts, and generated
answers out of runtime traces. The `slack-db-eval --langsmith` command runs the real graph against the versioned
seven-case dataset and attaches the deterministic and semantic quality gate to every example.

The final [`slack-db-bot-gpt-5.4-mini-gpt-5.4-mini-0b4a928b`](https://smith.langchain.com/o/b966086f-707d-4439-bdef-9e441ce1ceee/datasets/29233e7f-4918-473a-a608-b397859e763a/compare?selectedSessions=83b07b2b-1010-40d6-8fa3-b41a7154bab7)
experiment ran all seven assignment cases through the application and passed all seven quality gates. Detailed measured
results and limitations are recorded in [`EVALUATION.md`](EVALUATION.md).

![LangSmith evaluation showing the versioned examples, retrieval tool calls, and GPT-5.4 mini model spans](docs/images/langsmith-evaluation.png)

## Local setup

Prerequisites are Python 3.12+, [uv](https://docs.astral.sh/uv/), a Slack app installed in a test workspace, and an
OpenAI API key.

```bash
uv sync --all-groups
cp .env.example .env
```

Create or update the Slack app from [`slack-manifest.yaml`](slack-manifest.yaml). Generate an app-level token with
`connections:write`, install the app, invite it to the test channel, and fill the Slack IDs and tokens in `.env`.
`SLACK_TEST_CHANNEL_ID` must also appear in `SLACK_ALLOWED_CHANNEL_IDS`. Event Subscriptions must be enabled with
`app_mention` and `message.channels`; Socket Mode does not use a request URL.

The default database URL is pinned to a Git commit and verified with SHA-256 before use. To avoid downloading it, put
the same verified snapshot at `DATABASE_PATH`. The source is opened with SQLite `mode=ro`, `immutable=1`,
`query_only=ON`, and a deterministic SQL allowlist. Runtime state is written only under `.data/`.

Start the bot:

```bash
uv run slack-db-bot
```

Mention `@LangChain Bot` in an allowed public channel. Replies stay in the source thread. A plain thread follow-up is
accepted after the first mention. Direct messages are outside the assignment scope and are rejected. The bot
broadcasts a reply only when the request explicitly says to share or post it to the channel. Answers longer than
Slack's safe message size get a thread summary and Markdown attachment.

## Tests and evaluation

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
uv run slack-db-eval --output evaluation/results/latest.json
uv run slack-db-eval --langsmith
```

The test suite covers the SQL guard and immutable database, hybrid retrieval, deterministic grounding, prompt
injection, bounded conversation memory, event-ledger crash recovery, Slack authorization and engagement, ordered
follow-ups, deduplication, explicit broadcasting, dependency failure, and long-answer attachment delivery. The
packaged seven-case dataset is
[`src/slack_db_bot/data/acceptance.json`](src/slack_db_bot/data/acceptance.json); measured results and known limits are in
[`EVALUATION.md`](EVALUATION.md).

## Security and operations

Only configured workspace and channel IDs are accepted. The model receives one read-only search tool and no shell,
filesystem, MCP, arbitrary network, or Slack tools. SQL permits one bounded `SELECT` over allowlisted tables, columns,
and functions. Retrieved text is marked as untrusted, structured answers must map claims to retrieved evidence, inferences need
two sources, numeric details must occur in cited excerpts, and Slack delivery uses the authenticated source channel and
thread only.

Each turn has explicit model, tool, rewrite, Slack-call, output-token, queue, concurrency, and 120-second wall-clock
limits. OpenAI transport retries are disabled; one grounded repair is allowed. Slack calls use a 15-second timeout and
one bounded retry for short `429` responses. Event and conversation records expire after 30 days, old thread turns are
compacted into a bounded extractive summary, and an interrupted event claim becomes retryable after 10 minutes.

Logs contain event IDs, hashed thread keys, latency, calls, rewrites, and token counts; they omit prompts, evidence,
answers, users, and credentials. Set `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true` so traces retain
execution metadata without Slack text or database excerpts. Evaluation examples are synthetic.

Secrets belong only in `.env`, which is gitignored. `.data/`, semantic vectors, checkpoints, event history, and local
traces are also gitignored. Stop the process with Ctrl-C; the Socket Mode session and checkpoint connection close
cleanly.

## Design submission

[`DESIGN.md`](DESIGN.md) is the required human-authored explanation of the architecture and tradeoffs.
