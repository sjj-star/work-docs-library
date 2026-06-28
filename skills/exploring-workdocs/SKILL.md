---
name: exploring-workdocs
description: Use when answering technical questions, tracing source provenance, comparing concepts across documents, or building a structured report from the work-docs-library knowledge base and graph
---

# Exploring Work Docs

## Overview

Combine semantic search and the cross-document knowledge graph to answer technical questions. The plugin returns rich structured context (chunks + entities + relations + source documents) from each query tool, but it does **not** synthesize answers or perform smart routing. The Agent/Skill performs all intelligent dynamic analysis by composing the structured outputs of the atomic MCP tools.

## When to Use

- The user asks a technical question that may span multiple documents.
- The user wants to compare how two documents describe the same module, register, or concept.
- The user asks "where does this come from?" or "what is related to X?"
- The user needs a concise summary plus evidence, not just raw search results.

## Decision Tree: Which Tool?

```
User intent
├── Text / natural-language question
│   ├── Need highest recall or concept is fuzzy
│   │   └── search mode=semantic
│   ├── Need keyword + semantic balance
│   │   └── search mode=hybrid
│   ├── Need highest precision and can afford LLM cost
│   │   └── search mode=reranked
│   └── Complex multi-hop planned retrieval
│       └── agentic-search Skill (orchestrates multiple search/explore/read calls)
├── Named entity, relationship, path, provenance, or conflict
│   └── explore
│       ├── "Find entity X"              → mode=entity
│       ├── "What is connected to X?"    → mode=neighbors
│       ├── "Describe the X subgraph"    → mode=subgraph
│       ├── "How does X relate to Y?"    → mode=path
│       ├── "Where does X come from?"    → mode=provenance
│       └── "Do documents disagree on X?" → mode=conflicts
└── Read the original source text / block / chapter
    └── read
```

## Workflow

1. **Verify document state**
   - Call `mcp__workdocs__status` to confirm relevant documents are `done`.

2. **(Optional) Global quality check**
   - Call `mcp__workdocs__status` with `{"scope": "quality"}` to check for failed documents, unembedded blocks, or graph inconsistencies.

3. **Find relevant context**
   - For text questions, call `mcp__workdocs__search` with the appropriate `mode`:
     - `{"mode": "semantic", "text": "...", "top_k": 8}`
     - `{"mode": "hybrid", "text": "...", "top_k": 8}`
     - `{"mode": "reranked", "text": "...", "top_k": 5, "candidate_k": 20}`
   - Try synonyms and abbreviations if the first search returns poor results.

4. **Extract candidate entities**
   - From search results, identify entity names and types (Register, Module, Peripheral, etc.).

5. **Explore the graph**
   - For each candidate entity, call `mcp__workdocs__explore` with `{"mode": "neighbors", "entity_type": "...", "name": "...", "depth": 1}`.
   - If comparing two entities, call `mcp__workdocs__explore` with `{"mode": "path", "from_type": "...", "from_name": "...", "to_type": "...", "to_name": "..."}`.

6. **Trace provenance and conflicts**
   - Call `mcp__workdocs__explore` with `{"mode": "provenance", "entity_type": "...", "name": "..."}` for central entities.
   - Call `mcp__workdocs__explore` with `{"mode": "conflicts", "entity_type": "...", "name": "..."}` if documents disagree.

7. **Read source chunks**
   - Call `mcp__workdocs__read` with `{"doc_id": "...", "chapter": "..."}` or `{"chunk_db_id": N}`.

8. **Produce the structured report**
   - **Summary**: 2-4 sentences answering the question.
   - **Key entities**: type, name, properties, confidence, verified.
   - **Key relations**: from, to, type, properties.
   - **Source chunks**: doc_id, chapter_title, content_preview.
   - **Next steps**: more specific follow-up queries if information is incomplete.

## Parameter Strategy

- `search.top_k`: 5-10 for overview, 15-20 when the topic is broad.
- `search.candidate_k` (reranked mode): 3-4 × top_k.
- `explore` `neighbors`/`subgraph` depth: 1 for initial context, 2 only if relations are sparse.
- Prefer exact `name` in `explore`; if no result, try substring or wildcard.
- Filter out low-confidence (`confidence < 0.5`) or unverified results unless the user asks for everything.

## Red Flags

- Answering from `search` alone and ignoring the graph.
- Assuming two documents describe the same entity without checking source provenance.
- Calling `graph_upsert_*` or `graph_feedback` through MCP; they are admin-only.
- Returning raw tool output instead of a synthesized report.
- Expecting the plugin to choose `search.mode` or `explore.mode` automatically.
