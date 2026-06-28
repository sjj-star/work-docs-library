---
name: using-workdocs
description: Use when a user wants to ingest technical PDFs, query a document knowledge base, or explore cross-document relationships through the work-docs-library Kimi Code plugin
---

# Using Work Docs Library

## Overview

The work-docs-library plugin turns technical PDFs into a queryable knowledge base with vector search, hybrid retrieval, optional reranking, and a cross-document knowledge graph. This skill is the entry point. For concrete workflows, load the required sub-skill.

## When to Use

- The user mentions importing, adding, or updating PDFs or directories.
- The user asks a technical question like "What is the reset sequence of the SPI module?" or "Compare GPIO in doc A and doc B."
- The user wants to trace the source of a fact, find related registers, or explore concept relationships.
- The user reports missing entities, wrong relations, or wants to check conflict logs.

## How to Choose a Workflow

- **Import or update documents** → **REQUIRED SUB-SKILL:** `ingesting-workdocs`
- **Simple technical questions or single-document lookups** → **REQUIRED SUB-SKILL:** `exploring-workdocs`
- **Complex multi-hop questions spanning multiple documents or requiring graph traversal** → **REQUIRED SUB-SKILL:** `agentic-search` (user-level skill, located at `~/.agents/skills/agentic-search`). Load it before planning retrieval.
- **Produce a cited, structured answer from retrieved context** → **REQUIRED SUB-SKILL:** `synthesizing-workdocs`
- **Manual maintenance** (entity correction, reprocessing, feedback, global graph rebuild) → not exposed through MCP. Use `scripts/admin_tools.py`.

**Typical call chain for a complex question**:

```
using-workdocs (this skill)
  → agentic-search (plan multi-hop retrieval)
    → exploring-workdocs (execute each search/explore/read step)
    → synthesizing-workdocs (produce the final cited report)
```

## MCP Tools Quick Reference (5 tools)

| Tool | Purpose |
|------|---------|
| `mcp__workdocs__search` | Search the knowledge base. `mode=semantic` for vector search, `mode=hybrid` for BM25 + vector RRF, `mode=reranked` for hybrid + LLM cross-encoder reranking. |
| `mcp__workdocs__explore` | Explore the knowledge graph. `mode=entity` / `neighbors` / `subgraph` / `path` / `provenance` / `conflicts`. |
| `mcp__workdocs__read` | Read source content: a chapter, a content block, or a list of blocks by doc/chapter/concept. |
| `mcp__workdocs__ingest` | Import PDF(s) end-to-end. |
| `mcp__workdocs__status` | Status dashboard. `scope=overview/documents/vectors/graph/blocks/headings/conflicts/feedback/config/quality/ingest_pipeline/toc/all`. |

> **Note:** `config` and `toc` are now scopes of `status`, not standalone tools.

## Agent Cognitive Model

Map the user's intent to one of the five atomic tools:

- **Text question** → `search`
- **Entity / relationship / path / provenance / conflict** → `explore`
- **Read source evidence** → `read`
- **Ingest or update documents** → `ingest`
- **Check progress / config / TOC** → `status`

No LLM synthesis or smart routing happens inside the plugin. The Agent/Skill performs all intelligent dynamic analysis by calling the atomic tools and composing their structured outputs.

## Rules

- Do NOT call `graph_upsert_entity`, `graph_delete_entity`, `graph_upsert_relation`, `graph_delete_relation`, `graph_feedback`, `rebuild_global_graph`, or `reprocess` through MCP. They are not exposed as MCP tools; use `scripts/admin_tools.py` instead.
- Do NOT expect the plugin to synthesize answers or choose search modes for you. Select `search.mode` and `explore.mode` explicitly based on the user's question.
- Always confirm the ingest path with the user when unsure; all paths are sandboxed.
- For long-running `ingest` calls, use background tasks with a timeout of at least 1800 seconds and poll `status`.

## Common Pitfalls

- Assuming ingestion succeeds immediately. Always poll `status` after `ingest`.
- Searching with only one phrasing. Try synonyms and technical abbreviations.
- Trusting graph relations without tracing provenance via `explore` `mode=provenance`.
- Forgetting to check `explore` `mode=conflicts` when answers from different documents disagree.
- Using `search` when the user is asking about a named entity or relationship; prefer `explore` for entity-centric questions.
