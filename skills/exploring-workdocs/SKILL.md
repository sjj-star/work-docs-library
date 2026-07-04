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
│   ├── Default first attempt
│   │   └── search mode=hybrid
│   ├── Highest recall or concept is fuzzy
│   │   └── search mode=semantic
│   ├── Highest precision AND cost is acceptable
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

**Hard rule**: Unless the user explicitly asks for the highest precision or the answer quality is critical, start every text question with `search mode=hybrid`.

**Hard rule**: `search mode=reranked` costs an extra LLM call. Do not use it as the default.

## Workflow

### Standard Loop: search → explore → read → synthesize

1. **Verify document state**
   - Call `mcp__workdocs__status` to confirm relevant documents are `done`.

2. **(Optional) Global quality check**
   - Call `mcp__workdocs__status` with `{"scope": "quality"}` to check for failed documents, unembedded blocks, or graph inconsistencies.

3. **Find relevant context**
   - For text questions, call `mcp__workdocs__search` with `mode="hybrid"` first:
     - `{"mode": "hybrid", "text": "...", "top_k": 8, "include_graph": true}`
   - Use `include_graph=true` so the result includes entities and relations.
   - Try synonyms and abbreviations if the first search returns poor results.
   - Switch to `mode="semantic"` only if recall is too low.
   - Switch to `mode="reranked"` only if precision is critical and you can afford the extra LLM call.

4. **Extract candidate entities**
   - From `result.entities`, identify entity names and types (Register, Module, Peripheral, etc.).
   - Treat `result.relations` as the bridge to deeper exploration.

5. **Explore the graph**
   - For each candidate entity, call `mcp__workdocs__explore` with `{"mode": "neighbors", "entity_type": "...", "name": "...", "depth": 1}`.
   - If comparing two entities, call `mcp__workdocs__explore` with `{"mode": "path", "from_type": "...", "from_name": "...", "to_type": "...", "to_name": "..."}`.
   - For a broader view, use `{"mode": "subgraph", "depth": 1}`.

6. **Trace provenance and conflicts**
   - Call `mcp__workdocs__explore` with `{"mode": "provenance", "entity_type": "...", "name": "..."}` for central entities.
   - Call `mcp__workdocs__explore` with `{"mode": "conflicts", "entity_type": "...", "name": "..."}` if documents disagree.

7. **Read source chunks**
   - Call `mcp__workdocs__read` with `{"doc_id": "...", "chapter": "..."}` or `{"chunk_db_id": N}`.
   - Read at least one source block for every major claim in the final answer.

8. **Produce the structured report**
   - Use `synthesizing-workdocs` Skill for the exact report template.
   - Include: Summary / Key entities / Key relations / Source chunks / Citations / Next steps.

### Failure Fallbacks

- `search(mode="hybrid")` returns nothing → try `search(mode="semantic")` with synonyms.
- `explore(mode="entity")` returns nothing → the entity name may be wrong; try `explore(mode="neighbors")` on a related entity, or broad `search`.
- `explore(mode="path")` returns no path → increase `max_depth` by 1 once, then stop.
- Results disagree across documents → call `explore(mode="conflicts")` and report the discrepancy explicitly.

## Parameter Strategy

- `search.top_k`: 5-10 for overview, 15-20 when the topic is broad.
- `search.mode`: **default to `"hybrid"`**. Use `"semantic"` only for fuzzy recall. Use `"reranked"` only when precision is critical and cost is acceptable.
- `search.rerank_candidate_k` (reranked mode): 3-4 × top_k.
- `search.include_graph`: default `true` so you get entities and relations for the next explore step.
- `explore` `neighbors`/`subgraph` depth: 1 for initial context, 2 only if relations are sparse.
- Prefer exact `name` in `explore`; if no result, try substring or wildcard.
- Prefer `verified` entities/relations in the final answer. If you include unverified results, explicitly say so.

## Handling User Corrections

If the user says an entity, relation, or answer is wrong:

1. Load `fixing-workdocs`.
2. Use `status scope=trace` to replay the retrieval path.
3. Use `explore(mode=provenance)` and `read` to locate the source.
4. Generate the appropriate admin command and ask the user for confirmation.

## Red Flags — STOP and Correct

| Red flag | Why it is wrong |
|----------|-----------------|
| Using `search mode=reranked` as the first attempt | It costs an extra LLM call. Start with `hybrid`. |
| Answering from `search` alone and ignoring the graph | You miss entity/relationship context. |
| Assuming two documents describe the same entity without checking source provenance | Documents may disagree. |
| Calling `graph_upsert_*` or `graph_feedback` through MCP | They are admin-only, not MCP tools. |
| Returning raw tool output instead of a synthesized report | The user needs an answer, not JSON. |
| Expecting the plugin to choose `search.mode` or `explore.mode` automatically | The Agent must decide explicitly. |
| Calling old tool names like `semantic_search`, `search_hybrid`, `graph_query`, `query`, `get_content` | These tools were removed. Use `search`, `explore`, `read`. |

## Rationalization Table

| Excuse | Reality |
|--------|---------|
| "`reranked` is safer" | It is more expensive. `hybrid` is the safe default. |
| "I don't need `include_graph`" | You will miss entities and relations that drive the next `explore` step. |
| "One search is enough" | Technical questions usually require `search` + `explore` + `read`. |
| "The plugin should pick the right mode" | The plugin exposes atomic tools; mode selection is the Agent's job. |
