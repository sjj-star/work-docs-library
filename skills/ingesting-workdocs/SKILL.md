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
   - To view per-stage pipeline status, call `mcp__workdocs__status` with `{"scope": "pipeline"}`.

4. **Choose the action**
   - New PDF or directory → `mcp__workdocs__ingest` with `{"path": "..."}`.
   - Preview before ingesting → `mcp__workdocs__ingest` with `{"path": "...", "dry_run": true}` (parses and builds batches without calling LLM API).
   - Document already `done` but some content_blocks failed embedding (e.g., `vectors < blocks` in status) → **safe to re-run `mcp__workdocs__ingest`**. The pipeline will skip completed LLM extraction and entity ingestion; only missing embeddings are recomputed.
   - Document stuck in `processing` because LLM or Embedding API was missing → **safe to re-run `mcp__workdocs__ingest`**. The pipeline records per-stage status; previously `skipped` stages are retried automatically once the API is configured.
   - Document `failed` or you need a forced full re-run (e.g., changed prompts, fixed parser, suspect corrupted graph/vector state) → **admin-only**: run `python scripts/admin_tools.py reprocess --params '{"doc_id": "..."}'`. This is not exposed as an MCP tool because it removes and rebuilds graph/vector contributions.
   - Fine-grained stage-by-stage control (e.g., only re-run Stage 6 embedding) → **admin-only**: use `python scripts/admin_tools.py stage6_submit_embed_batches --params '{"doc_id": "..."}'`.

## Continuation and Recovery

The `ingest` MCP tool is idempotent for already-completed stages. When you call it again on an existing document:

- Stage 1 re-parses the PDF (small overhead).
- Stage 2 rebuilds the JSONL.
- Stage 3 is skipped if LLM API is not configured; otherwise retries failed/missing Chat requests.
- Stage 4 writes content_blocks (entities may be empty if stage3 was skipped).
- Stage 5 rebuilds the embedding JSONL, including only blocks without embeddings.
- Stage 6 is skipped if Embedding API is not configured; otherwise retries embedding for missing blocks only.

Document-level status remains `processing` until all non-skipped stages are `done`. This makes `ingest` the correct MCP-level recovery tool for:

- Transient Embedding API failures (`status` shows `failed_blocks > 0` but document is `done`).
- Transient Chat API timeouts in Chat mode.
- LLM/Embedding API was missing during a previous run (`status` shows `processing` with `skipped` stages in `scope=pipeline`).

It is NOT suitable for:

- Forcing a full re-extraction (use `reprocess`).
- Skipping the PDF re-parse for large documents (use admin `stage6_submit_embed_batches` instead).
- Fixing corrupted entity/relation data (use `reprocess`).

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
- Re-ingesting a document that is already `done` **without first checking whether any blocks are still failed**.
- Ignoring a `failed` status or retrying indefinitely without checking logs.
- Using absolute paths outside the allowed directories.
- Confusing `ingest` (safe resume) with `reprocess` (forced full rebuild). Use `reprocess` only when graph/vector state is suspect.
- Assuming a document is incomplete just because its status is `processing` — check `scope=pipeline` to see which stages are `skipped` or `failed`.
