Goal: We need to build a Slack Q\&A bot that is grounded in the supplied SQLite database. A user should be able to tag (@bot) the bot in the channel, receive a response in the original thread, and continue the conversation with some follow-up questions. The system must be able to answer direct lookups, account summaries, comparisons, and any other questions grounded within the database.

The main quality goals are correctness and completeness. Additionally, our constraints are firstly security and secondly action efficiency. The bot should be able to expose the evidence behind an answer, tell facts from inferred ones, share uncertainty and/or uncertainties and not say anything if the database can’t support the claim.

The scope should be a strong local Slack demo with reliability and security as non-negotiables. This uses 1 local Python process and Slack socket mode. We use GPT-5.4-mini for the model in the final configuration.

Architecture: The architecture is based on an explicit LangGraph StateGraph. We’ll have deterministic nodes to own the normalization, authorization, budgets, retrieval/output validation, and any Slack side effects. This process should have an async Slack adapter, per-thread work queue, compiled graph, retrieval service built in, and local persistence. A deep agent specializes in handling questions that require multiple sources for the investigation. It’ll receive inputs, approved tools, and the action budget for it to return an evidence report. The StateGraph has full control over the lifecycle, security, validation, and any other side effects. LangGraph’s explicit workflow for routing, retrieval, validation, and delivery is key, with a dedicated specialist for when multiple sources require investigation.

Request Flow:

The request flow should be 

1. Slack receives a socket mode event and acknowledges it promptly  
2. When is @bot mentioned, the conversation begins with later messages triggering the bot only inside an already engaged thread  
3. Event recorded in the deduplication database and placed in the queue for the thread  
4. A routing node identifies the question type and creates a retrieval plan, keeping in mind the bounded limits  
5. It’ll use the retrieval layer combining structured SQL, FTS5 search, and local semantic retrieval (only pulling from each when needed)  
6. LangGraph will make sure the retrieved evidence is relevant and sufficient, and attempt at most 2 query rewrites and 8 retrieval/tool calls  
7. A synthesis node creates an answer containing any claims, evidence IDs, fact labels, uncertainty, and conflict notes  
8. The deterministic validator rejects any unsupported claims, invented citations, bad output, or unauthorized delivery behavior  
9. Slack bot posts one response in the original thread 

State/Memory: The graph state includes the Slack request context, recent messages, rolling conversation summary, evidence, previous answers, and delivery status. The request identity should be immutable for a turn since messages are appended.

Long threads should retain recent exact turns \+ a rolling summary with unresolved questions, previously cited/solved Q\&A, and evidence. 

Database \+ Retrieval: The given database is going to be in read-only mode. The retrieval layer is going to use structured SQL for exact filters/comparisons, FTS5 for keyword/phrase matching, and a small local semantic index for questions where wording differs from the source. Then the results are combined and re-ranked based on entity matches, source status, and date.

Answer format: The model should return a structured answer with response, evidence IDs, fact labels, uncertainty, and any abstaining results. This Slack response should lead with the answer and include compact source references. 

Security: Slack messages, search results, database rows, summaries, and Deep Agents outputs should be treated as untrusted data. While they can provide evidence, they can’t override system rules or get access to more tools.

This is enforced via deterministic code. The model won’t have access to the host filesystem, shell, or any arbitrary network calls/SQL.

Evaluation: We’ll require the 7 assignment questions to be accepted, with each question having a defined rubric, and they should be reviewed by a human.

Results: The final submission should have answer correctness, completeness, citation quality, latency time, reliability, token usage, and tool call count. 