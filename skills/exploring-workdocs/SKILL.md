---
name: exploring-workdocs
description: Use when answering technical questions, tracing source provenance, comparing concepts across documents, or building a structured report from the work-docs-library knowledge base and graph
---

# Exploring Work Docs

## Overview

Combine semantic search and the cross-document knowledge graph to answer technical questions. The default output is a structured report that shows the answer, key entities, key relations, and source chunks so the user can verify the knowledge chain.

## When to Use

- The user asks a technical question that may span multiple documents.
- The user wants to compare how two documents describe the same module, register, or concept.
- The user asks "where does this come from?" or "what is related to X?"
- The user needs a concise summary plus evidence, not just raw search results.

## Workflow

1. **Verify document state**
   - Call `mcp__workdocs__status` to confirm relevant documents are `done`.

2. **Semantic search**
   - Call `mcp__workdocs__semantic_search` with `{"text": "...", "top_k": 8, "graph_depth": 1}`.
   - Try synonyms and abbreviations if the first search returns poor results.

3. **Extract candidate entities**
   - From search results, identify entity names and types (Register, Module, Peripheral, etc.).

4. **Graph lookup**
   - For each candidate entity, call `mcp__workdocs__graph_query` with `{"entity_type": "...", "name": "...", "depth": 1}`.
   - If comparing two entities, call `mcp__workdocs__graph_path` with `{"from_type": "...", "from_name": "...", "to_type": "...", "to_name": "..."}`.

5. **Trace provenance and conflicts**
   - Call `mcp__workdocs__graph_provenance` with `{"entity_type": "...", "name": "..."}` for central entities.
   - Call `mcp__workdocs__graph_conflicts` with `{"entity_type": "...", "name": "..."}` if documents disagree.

6. **Read source chunks**
   - Call `mcp__workdocs__get_content` with `{"doc_id": "...", "chapter": "..."}` or `{"chunk_db_id": N}`.

7. **Produce the structured report**
   - **摘要**：2-4 sentences answering the question.
   - **关键实体**：type, name, properties, confidence, verified.
   - **关键关系**：from, to, type, properties.
   - **来源块**：doc_id, chapter_title, content_preview.
   - **后续建议**：more specific follow-up queries if information is incomplete.

## Parameter Strategy

- `semantic_search.top_k`: 5-10 for overview, 15-20 when the topic is broad.
- `semantic_search.graph_depth`: 1 for initial context, 2 only if relations are sparse.
- `graph_query`: prefer exact `name`; if no result, try substring or wildcard.
- Filter out low-confidence (`confidence < 0.5`) or unverified results unless the user asks for everything.

## Red Flags

- Answering from semantic search alone and ignoring the graph.
- Assuming two documents describe the same entity without checking source provenance.
- Calling `graph_upsert_*` or `graph_feedback` through MCP; they are admin-only.
- Returning raw tool output instead of a synthesized report.
