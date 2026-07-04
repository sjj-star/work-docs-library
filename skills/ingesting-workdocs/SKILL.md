---
name: ingesting-workdocs
description: Use when importing or updating technical PDFs and directories into the work-docs-library knowledge base
---

# Ingesting Work Docs

## Overview

Ingestion parses PDFs, extracts chapters, builds LLM batches, runs entity/relation extraction, and writes both the graph and vector index. It may take minutes to hours for large documents because it involves PDF parsing and LLM Batch API calls.

## When to Use

- The user provides a PDF file or directory to add to the knowledge base.
- The user wants to update a document that has already been ingested.
- The user reports that a previous ingestion failed or produced incomplete results.

## Workflow

1. **Check configuration**
   - Call `mcp__workdocs__status` with `{"scope": "config"}` to confirm LLM and embedding are configured.

2. **Confirm the path**
   - Resolve the user-provided path relative to the current working directory.
   - Confirm with the user if the path is ambiguous or outside the allowed sandbox.

3. **Check existing state**
   - Call `mcp__workdocs__status` (no params) to list existing documents.
   - If you already know the doc_id, call `mcp__workdocs__status` with `{"doc_id": "..."}` to see whether it is `done`, `processing`, or `failed`.
   - To view a document outline, call `mcp__workdocs__status` with `{"doc_id": "...", "scope": "toc"}`.

4. **Choose the action**
   - New PDF or directory → `mcp__workdocs__ingest` with `{"path": "..."}`.
   - Preview before ingesting → `mcp__workdocs__ingest` with `{"path": "...", "dry_run": true}` (parses and builds batches without calling LLM API).
   - Update/retry an existing document or a failed document → **admin-only**: run `python scripts/admin_tools.py reprocess --params '{"doc_id": "..."}'`. This is not exposed as an MCP tool because it may rewrite graph/vector state.
   - Fine-grained stage-by-stage control (rarely needed) → **admin-only**: use `python scripts/admin_tools.py stage1_parse ... stage6_submit_embed_batches`. These are not MCP tools.

5. **Run `ingest` as a background task**
   - **REQUIRED:** Call `mcp__workdocs__ingest` as a background task with `timeout` set to at least 1800 seconds.
   - The call returns a list of `doc_ids`; record all of them.

6. **Poll until completion**
   - Call `mcp__workdocs__status` with `{"doc_id": "..."}` every 30-60 seconds for each returned doc_id.
   - Stop when every document status is `done` or `failed`.

7. **Summarize results**
   - Report document id, status, number of chapters/blocks, entities, and relations.
   - If the status is `failed`, show the failure reason and suggest running `python scripts/admin_tools.py reprocess` (admin-only) after confirming with the user.

## Output Format

```
导入结果摘要
- 文档 ID: <doc_id>
- 状态: done / failed
- 章节数: N
- content_blocks: N
- 实体数: N
- 关系数: N
- 失败原因（如有）: ...
```

## Concrete Examples

```json
// Preview before ingesting (no API cost)
{"path": "./docs/spi.pdf", "dry_run": true}

// Start ingestion
{"path": "./docs/spi.pdf"}

// Poll progress
{"doc_id": "spi"}

// View TOC
{"doc_id": "spi", "scope": "toc"}

// Check config
{"scope": "config"}
```

## Red Flags

- Calling ingest synchronously and assuming it finishes quickly.
- Re-ingesting a document that is already `done` without asking the user.
- Ignoring a `failed` status or retrying indefinitely without checking logs.
- Using absolute paths outside the allowed directories.
