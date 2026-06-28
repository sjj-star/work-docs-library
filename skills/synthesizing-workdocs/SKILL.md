---
name: synthesizing-workdocs
description: Use when you have retrieved structured context from work-docs-library and need to produce a cited, structured technical answer or report
---

# Synthesizing Work Docs

## Overview

The work-docs-library plugin returns rich structured context: chunks, entities, relations, source documents, paths, provenance, and conflicts. This skill turns that raw context into a concise, cited technical answer.

## When to Use

- You have just called `mcp__workdocs__search`, `mcp__workdocs__explore`, or `mcp__workdocs__read`.
- You need to answer a user question with evidence, not just return raw tool output.
- You need to compare, summarize, or resolve conflicts across multiple sources.

## When NOT to Use

- You have not yet retrieved any context. Use `exploring-workdocs` or `agentic-search` first.
- The user only wants raw search results.

## Standard Report Structure

Always produce the answer in this order:

```markdown
## Summary
2-4 sentences that directly answer the question.

## Key Entities
| Type | Name | Key Properties | Confidence | Verified |
|------|------|----------------|------------|----------|
| ...  | ...  | ...            | ...        | ...      |

## Key Relations
| Relation | From | To | Properties |
|----------|------|----|-----------|
| ...      | ...  | ...| ...       |

## Source Evidence
| doc_id | chapter_title | content_preview |
|--------|--------------|------------------|
| ...    | ...          | ...              |

## Citations
[doc_id, chapter_title]

## Conflicts or Uncertainties
- If documents disagree, state the disagreement and which source says what.
- If information is missing, say so.

## Next Steps
- Specific follow-up queries if the answer is incomplete.
```

## How to Build Each Section

### Summary
- Answer the user's exact question first.
- Use technical terms from the retrieved context.
- Do not introduce information not present in the context.

### Key Entities
- Pull from `result.entities`.
- Select entities that are central to the answer.
- Include `confidence` and `verified` to signal reliability.

### Key Relations
- Pull from `result.relations`.
- Use these to explain "how X relates to Y".

### Source Evidence
- Pull from `result.chunks`.
- Include the most relevant blocks, especially those used for direct claims.

### Citations
- Format: `[doc_id, chapter_title]`.
- Every factual claim must have at least one citation.
- If a claim relies on multiple blocks, use multiple citations.

### Conflicts or Uncertainties
- If you called `explore(mode="conflicts")`, report discrepancies explicitly.
- Do not hide uncertainty. Say "the knowledge base does not contain X" if true.

## Rules

- Do not invent facts. Every claim must trace to a retrieved chunk, entity, or relation.
- Do not copy large blocks of text. Paraphrase and cite.
- Do not omit citations to save space.
- If confidence is low or the entity is unverified, say so.

## Red Flags

| Red flag | Why it is wrong |
|----------|-----------------|
| Returning raw JSON from the tool | The user asked a question, not for raw data. |
| No citations | The answer is not verifiable. |
| Claims without a matching chunk/entity/relation | You invented information. |
| Ignoring conflicts | You present uncertain information as certain. |
