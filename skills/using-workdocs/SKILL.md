---
name: using-workdocs
description: Use when a user wants to ingest technical PDFs, query a document knowledge base, or explore cross-document relationships through the work-docs-library Kimi Code plugin
---

# Using Work Docs Library

## Overview

The work-docs-library plugin turns technical PDFs into a queryable knowledge base with vector search and a cross-document knowledge graph. This skill is the entry point. For concrete workflows, load the required sub-skill.

## When to Use

- The user mentions importing, adding, updating, or reprocessing PDFs or directories.
- The user asks a technical question like "What is the reset sequence of the SPI module?" or "Compare GPIO in doc A and doc B."
- The user wants to trace the source of a fact, find related registers, or visualize concept relationships.
- The user reports missing entities, wrong relations, or wants to check conflict logs.

## How to Choose a Workflow

- **Import or update documents** → **REQUIRED SUB-SKILL:** `ingesting-workdocs`
- **Answer technical questions, compare concepts, trace provenance** → **REQUIRED SUB-SKILL:** `exploring-workdocs`
- **Manual maintenance** (entity correction, feedback, global graph rebuild) → not exposed through MCP. Use `scripts/admin_tools.py`.

## MCP Tools Quick Reference

| Tool | Purpose |
|------|---------|
| `mcp__workdocs__ingest` | Import PDF(s) end-to-end |
| `mcp__workdocs__reprocess` | Re-run pipeline for a doc or failed docs |
| `mcp__workdocs__semantic_search` | Vector search; set `graph_depth>0` to include graph |
| `mcp__workdocs__query` | Find content_blocks by doc/chapter/concept |
| `mcp__workdocs__get_content` | Read a chapter or block raw content |
| `mcp__workdocs__status` | List docs or see progress |
| `mcp__workdocs__toc` | Show document outline |
| `mcp__workdocs__graph_query` | Query entity / neighbors / subgraph |
| `mcp__workdocs__graph_path` | Find path between two entities |
| `mcp__workdocs__graph_provenance` | Trace entity to source doc/chunk |
| `mcp__workdocs__graph_conflicts` | View property conflict logs |
| `mcp__workdocs__config` | Show effective config (masked) |

## Rules

- Do NOT call `graph_upsert_entity`, `graph_delete_entity`, `graph_upsert_relation`, `graph_delete_relation`, `graph_feedback`, or `rebuild_global_graph` through MCP. They are not exposed.
- Always confirm the ingest path with the user when unsure; all paths are sandboxed.
- For long-running ingest/reprocess calls, use background tasks with a timeout of at least 1800 seconds and poll `status`.

## Common Pitfalls

- Assuming ingestion succeeds immediately. Always poll `status` after ingest/reprocess.
- Searching with only one phrasing. Try synonyms and technical abbreviations.
- Trusting graph relations without calling `graph_provenance`.
- Forgetting to check `graph_conflicts` when answers from different documents disagree.
