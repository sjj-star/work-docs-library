---
name: fixing-workdocs
description: Use when a user reports that a knowledge base answer, entity, or relation is wrong and you need to locate the source and propose a correction
---

# Fixing Work Docs

## Overview

This skill closes the loop between "user finds a problem" and "knowledge base gets corrected". It uses the usage trace and provenance tools to pinpoint where a wrong entity or relation entered the answer, then proposes a concrete fix for user confirmation.

## When to Use

- The user says an answer is wrong, incomplete, or contains hallucinated entities/relations.
- The user points to a specific entity or relation and says it is incorrect.
- You want to mark an entity/relation as problematic so it can be reviewed later.

## When NOT to Use

- The user is simply asking a new question. Use `exploring-workdocs` or `agentic-search` instead.
- You need to modify the knowledge base without user confirmation. All fixes here require explicit approval.

## Core Workflow

```
1. Replay
   status scope=trace [session_id=...]
   → see which search/explore/read calls produced the answer

2. Locate source
   explore(mode=provenance, entity_type=..., name=...)
   → find the source blocks that mention the problematic entity/relation

3. Read evidence
   read(chunk_db_id=...) or read(doc_id=..., chapter=...)
   → confirm the original text

4. Choose fix
   A. Flag for review      → graph_feedback rating=-1
   B. Update properties    → graph_upsert_entity / graph_upsert_relation
   C. Delete entirely      → graph_delete_entity / graph_delete_relation

5. Present to user
   → show the evidence, the proposed command, and the expected outcome

6. Execute (after user approval)
   → run the corresponding scripts/admin_tools.py command

7. Verify
   → explore(mode=entity, ...) to confirm the fix
   → optionally verify_entity / verify_relation to mark it as checked
```

## How to Identify the Problem Object

Ask the user:
- "Which entity or relation is wrong?"
- "What should the correct information be?"
- "Do you want to flag it, update it, or delete it?"

If the user is vague, use `status scope=trace` to show the last retrieval path and ask them to point to the step that went wrong.

## Fix Commands

These are **admin-only** commands, not MCP tools. Generate them exactly and ask the user to confirm before running.

### Flag as problematic

```bash
python scripts/admin_tools.py graph_feedback --params '{
  "rating": -1,
  "entity_type": "Module",
  "entity_name": "EPWM_TZ",
  "comment": "description does not match source text"
}'
```

### Update entity

```bash
python scripts/admin_tools.py graph_upsert_entity --params '{
  "entity_type": "Module",
  "name": "EPWM_TZ",
  "properties": {"description": "Trip-zone submodule for ePWM"},
  "verified": true
}'
```

### Delete entity

```bash
python scripts/admin_tools.py graph_delete_entity --params '{
  "entity_type": "Module",
  "name": "EPWM_TZ"
}'
```

### Delete relation

```bash
python scripts/admin_tools.py graph_delete_relation --params '{
  "rel_type": "TRIGGERS",
  "from_type": "Module",
  "from_name": "EPWM_TZ",
  "to_type": "Task",
  "to_name": "CLA1TASK1"
}'
```

## Rules

- **Never modify the knowledge base without user approval.**
- Always read the source block before proposing an update or deletion.
- If the source text supports the current entity/relation, explain that to the user and ask if the source document itself is wrong.
- If the source text contradicts the entity/relation, propose an update or deletion.
- After a fix, run `explore(mode=entity, ...)` to confirm.

## Red Flags

| Red flag | Why it is wrong |
|----------|-----------------|
| Calling `graph_upsert_*` or `graph_delete_*` as an MCP tool | They are admin-only, not in the MCP tool surface. |
| Fixing without reading the source block | You may overwrite correct information. |
| Deleting an entity that is connected to many relations | Check impact first; deleting an entity cascades to its edges. |
| Ignoring the user's correction | The user's domain knowledge is the ground truth. |
