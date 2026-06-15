---
name: using-workdocs
description: Use when working with the work-docs-library plugin to ingest technical PDFs, run semantic searches, or query the knowledge graph through Kimi Code MCP tools
---

# Using Work Docs Library

## Overview

This plugin turns technical PDFs into a queryable knowledge base with vector search and a cross-document knowledge graph.

## When to Use

- Ingest PDFs or directories of PDFs into the knowledge base.
- Search documents by meaning or by chapter/concept.
- Explore the knowledge graph: find entities, paths, provenance, or conflict logs.

## MCP Tools Quick Reference

| Tool | Purpose |
|------|---------|
| `mcp__workdocs__ingest` | Import PDF(s) end-to-end |
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

## Common Workflows

**Ingest and search**

1. `mcp__workdocs__ingest` with `{"path": "path/to/pdf_or_dir"}`
2. `mcp__workdocs__semantic_search` with `{"text": "reset sequence of the SPI module"}`

**Explore the graph**

1. `mcp__workdocs__graph_query` with `{"entity_type": "Register", "name": "SPICCR"}`
2. `mcp__workdocs__graph_provenance` with `{"entity_type": "Register", "name": "SPICCR"}`

## Rules

- Do NOT call `graph_upsert_entity`, `graph_delete_entity`, `graph_upsert_relation`, `graph_delete_relation`, `graph_feedback`, `reprocess`, or `rebuild_global_graph`; these are not exposed through MCP. Use `scripts/admin_tools.py` for manual maintenance.
- Confirm the path with the user before ingesting if you are unsure.
- All file paths are sandboxed; absolute paths outside allowed directories are rejected.
