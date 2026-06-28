---
name: agentic-search
description: Use when a user question requires multi-hop retrieval, cross-document synthesis, or structured graph traversal through the work-docs-library knowledge base
---

# Agentic Search over work-docs-library

## Overview

This skill guides multi-hop retrieval over a technical PDF knowledge base using only the 5 atomic MCP tools exposed by the work-docs-library plugin. The plugin returns structured context (chunks + entities + relations); this skill provides the planning and orchestration that turns that context into a cited answer.

## When to Use

- The question has two or more sub-questions that cannot be answered by a single search.
- The answer requires following entity → relation → entity chains (registers, modules, signals, peripherals).
- The user asks for comparison across documents or chapters.
- Single-shot `search` returns incomplete, noisy, or contradictory results.
- The answer must include precise source citations.

## When NOT to Use

- Simple factual lookup that a single `search` or `explore` can answer.
- Questions that only require reading one chapter (`read` is enough).
- Maintenance operations (entity correction, feedback, reprocessing). Use `scripts/admin_tools.py` instead.

## Core Pattern

Do NOT call `agentic_plan` as an MCP tool. Planning happens inside this skill. Follow this loop:

```
1. Plan
   Break the question into 2-5 concrete retrieval steps.
   Each step maps to ONE atomic MCP tool call.

2. Execute
   Run steps in order. Use previous results to refine later steps.

3. Read Evidence
   For high-relevance blocks, call `read` to get full text for citations.

4. Synthesize
   Combine chunks, entities, and relations into a structured answer with citations.
```

## MCP Tool Mapping

| SearchStep type | MCP tool | Notes |
|-----------------|----------|-------|
| `semantic` | `mcp__workdocs__search(mode="semantic")` | Fuzzy concept recall |
| `hybrid` | `mcp__workdocs__search(mode="hybrid")` | **Default first attempt** |
| `reranked` | `mcp__workdocs__search(mode="reranked")` | Precision-critical only; costs extra LLM call |
| `entity` | `mcp__workdocs__explore(mode="entity")` | Exact entity lookup |
| `neighbors` | `mcp__workdocs__explore(mode="neighbors")` | What is connected to X |
| `subgraph` | `mcp__workdocs__explore(mode="subgraph")` | Local graph around X |
| `path` | `mcp__workdocs__explore(mode="path")` | How X relates to Y |
| `provenance` | `mcp__workdocs__explore(mode="provenance")` | Where does X come from |
| `conflicts` | `mcp__workdocs__explore(mode="conflicts")` | Cross-document disagreements |
| `chapter` / `content` | `mcp__workdocs__read` | Read full source text |

## Recommended Multi-Hop Workflow

1. **Start with `search(mode="hybrid")`** on the core terms.
2. **Extract candidate entities** from `result.entities`.
3. **Expand each candidate** with `explore(mode="neighbors", depth=1)` or `explore(mode="subgraph", depth=1)`.
4. **Trace provenance** for central entities with `explore(mode="provenance")`.
5. **Compare documents** when needed with `explore(mode="conflicts")`.
6. **Read key blocks** with `read(chunk_db_id=...)` for exact quotes and citations.
7. **Synthesize** the answer.

## Example

**Question**: "How does the C28x CLA handle EPWM safety trip events, and what registers are involved?"

**Plan & Execute**:

```
search(mode="hybrid", text="C28x CLA EPWM safety trip registers", top_k=10)
→ entities: [Module EPWM_TZ, Task CLA1TASK1]

explore(mode="neighbors", entity_type="Module", name="EPWM_TZ", depth=1)
→ relations: EPWM_TZ TRIGGERS CLA1TASK1

explore(mode="provenance", entity_type="Module", name="EPWM_TZ")
→ source chunks

read(doc_id="spruh18", chapter="ePWM Trip-Zone Submodule")
read(doc_id="spruh18", chapter="CLA Task Trigger Sources")
```

**Synthesize**:

The C28x CLA handles EPWM safety trips via the ePWM Trip-Zone submodule (`EPWM_TZ`). When a trip condition occurs, `EPWM_TZ` forces the EPWM outputs to a safe state and can trigger `CLA1TASK1` through the trip-zone interrupt [spruh18, ePWM Trip-Zone Submodule][spruh18, CLA Task Trigger Sources]. Key registers include `TZSEL` (trip source selection), `TZCTL` (trip action), and `CLA1TASKSRCSEL` (CLA task routing).

## Failure Handling

- If `search` returns no relevant entities: broaden to `mode="semantic"` or try synonyms/abbreviations.
- If `explore(mode="neighbors")` returns empty: the entity may be misspelled; try `explore(mode="entity", name_pattern=...)`.
- If one step returns empty: do not stop. Continue with the remaining steps and note the gap in the final answer.
- If the plan becomes unclear after initial results: replan with at most 2 additional steps, then synthesize.

## Handling User Corrections

If the user challenges any entity or relation in your synthesized answer:

1. Load `fixing-workdocs`.
2. Use `status scope=trace` to see which step activated the problematic object.
3. Use `explore(mode=provenance)` to find its source blocks.
4. Propose a fix (flag/update/delete) and ask the user to confirm before running any admin command.

## Red Flags — STOP and Replan

| Red flag | Why it is wrong |
|----------|-----------------|
| Calling `agentic_plan` as an MCP tool | This tool was removed. Planning belongs in this skill. |
| Calling `semantic_search`, `search_hybrid`, `search_reranked`, `graph_query`, `graph_path`, `query`, or `get_content` | These names no longer exist. Use `search`, `explore`, or `read`. |
| Using `reranked` as the default first search | It costs an extra LLM call. Start with `hybrid`. |
| Ignoring `result.entities` and `result.relations` | These are the bridge to `explore`. |
| Synthesizing without reading any source block | Citations must point to actual retrieved evidence. |

## Rationalization Table

| Excuse | Reality |
|--------|---------|
| "`agentic_plan` used to work" | It was removed. The current plugin only exposes 5 tools. |
| "`reranked` is safer" | It is more expensive. Use `hybrid` first, then `reranked` only if precision is critical. |
| "I can just call `search` repeatedly" | Multi-hop questions need `explore` to follow entity/relationship chains. |
| "The plugin should synthesize for me" | The plugin returns structured context; synthesis is the Agent's job. |
