#!/usr/bin/env python3
"""
Agent batch helper for work-docs-library.

Provides a stable workflow for Agent to:
1. List pending (embedded-but-unsummarized) chunks.
2. Dump a batch of pending chunks to a text file for Agent reading.
3. Apply a batch of summaries/keywords from a JSON file back to the DB.
4. Filter low-value chunks automatically.
5. Show summarization progress.
6. Auto-pipeline with checkpoint/resume for long documents.

Usage:
  python agent_batch_helper.py list --doc-id <hash>
  python agent_batch_helper.py dump --doc-id <hash> --batch-size 20 --offset 0 --output batch.txt
  python agent_batch_helper.py apply --input batch.json
  python agent_batch_helper.py filter --doc-id <hash>
  python agent_batch_helper.py progress --doc-id <hash>
  python agent_batch_helper.py auto --doc-id <hash> --output-dir ./auto_batches --filter

JSON format for apply:
[
  {"chunk_db_id": 391, "summary": "...", "keywords": "a,b,c"},
  ...
]
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

_SKILL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_ROOT))

from core.config import Config
from core.db import KnowledgeDB

logger = logging.getLogger("agent_batch_helper")

_FILTER_CONFIG_PATH = _SKILL_ROOT / "prompts" / "filter_config.json"


def _load_filter_config():
    if _FILTER_CONFIG_PATH.exists():
        with open(_FILTER_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_FILTER_CONFIG = _load_filter_config()


def _validate_inside_skill(path: Path) -> bool:
    skill_root = Config.DB_PATH.parent.parent.resolve()
    try:
        path.resolve().relative_to(skill_root)
        return True
    except ValueError:
        return False


def _is_low_value(content: str, chapter_title: str, chunk_type: str = "text") -> bool:
    """Heuristic to skip packaging tables, disclaimers, TOC pages, and pure ASCII art.
    Rules are loaded from prompts/filter_config.json.
    """
    c = content.strip()
    ct = chapter_title.lower()
    cl = c.lower()
    cfg = _FILTER_CONFIG
    always = cfg.get("always_skip", {})
    heur = cfg.get("heuristic_skip", {})

    # Chunk-type filter (e.g. image_desc placeholders when no vision analysis is available)
    if chunk_type in always.get("chunk_types", []):
        return True

    # Minimum content length
    if len(c) < heur.get("min_content_length", 40):
        return True

    # Content keywords (copyright, disclaimer, etc.) — only match within the prefix
    # to avoid skipping normal pages that merely have a legal footer.
    prefix_ratio = always.get("content_prefix_ratio")
    if prefix_ratio is not None:
        prefix_len = int(len(cl) * prefix_ratio)
        min_chars = always.get("content_prefix_min_chars")
        max_chars = always.get("content_prefix_max_chars")
        if min_chars is not None:
            prefix_len = max(prefix_len, min_chars)
        if max_chars is not None:
            prefix_len = min(prefix_len, max_chars)
        search_window = cl[:prefix_len]
    else:
        prefix_len = always.get("content_prefix_chars")
        search_window = cl if prefix_len is None else cl[:prefix_len]
    for kw in always.get("content_keywords", []):
        if kw.lower() in search_window:
            return True

    # Chapter keywords (TOC, references, etc.)
    for kw in always.get("chapter_keywords", []):
        if kw.lower() in ct:
            return True

    # Dimension-page heuristic
    dim_cfg = heur.get("dimension_page", {})
    dim_chapters = [kw.lower() for kw in dim_cfg.get("chapter_keywords", [])]
    if any(kw in ct for kw in dim_chapters):
        indicators = dim_cfg.get("content_indicators", {})
        meets = True
        for ind, threshold in indicators.items():
            if cl.count(ind) < threshold:
                meets = False
                break
        if meets:
            exclusions = [kw.lower() for kw in dim_cfg.get("exclude_if_contains", [])]
            if not any(kw in cl for kw in exclusions):
                return True

    # Pure ASCII art
    art_chars = set("│─┌┐└┘├┤┬┴┼| -+=")
    non_ws = [ch for ch in c if not ch.isspace()]
    ratio = heur.get("ascii_art_ratio", 0.85)
    if non_ws and sum(1 for ch in non_ws if ch in art_chars) / len(non_ws) > ratio:
        return True

    return False


def _smart_batch(rows, target_chars: int = 25000, max_chunks: int = 12, min_chunks: int = 3):
    """Group rows into content-aware batches."""
    batches = []
    current = []
    current_chars = 0

    for row in rows:
        db_id, doc_id, chunk_id, content, ctype, ps, pe, ch_title = row
        # Start new batch if approaching limits and current batch is viable
        if current and (current_chars + len(content) > target_chars or len(current) >= max_chunks):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += len(content)

    if current:
        # Merge orphan tail into previous batch if too small
        if len(current) < min_chunks and batches:
            batches[-1].extend(current)
        else:
            batches.append(current)

    return batches


def _enrich_batch_with_images(batch, db):
    """Append image path hints for Agent vision analysis (scenario B)."""
    enriched = []
    for row in batch:
        db_id, doc_id, chunk_id, content, ctype, ps, pe, ch_title = row
        ck = db.get_chunk_by_db_id(db_id)
        if ck and ck.metadata.get("images"):
            # Only prompt Agent if script-layer vision has not yet processed these images
            needs_agent_vision = not all("vision_desc" in img for img in ck.metadata["images"])
            if needs_agent_vision:
                content += (
                    "\n\n[AGENT VISION REQUIRED: The following images are on this page. "
                    "Please read them and incorporate insights into the summary.]\n"
                )
                for img in ck.metadata["images"]:
                    content += f"- {img['path']}\n"
        enriched.append((db_id, doc_id, chunk_id, content, ctype, ps, pe, ch_title))
    return enriched


def _write_batch_txt(out_path: Path, batch):
    lines = []
    for db_id, doc_id, chunk_id, content, ctype, ps, pe, ch_title in batch:
        lines.append(f"--- CHUNK_DB_ID={db_id} | {chunk_id} | {ch_title} P{ps} ---")
        lines.append(content)
        lines.append("\n" + "=" * 80 + "\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _apply_json_file(json_path: Path) -> int:
    """Internal apply logic; returns number of updated chunks."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Invalid JSON | %s must be a list of objects", json_path.name)
        return 0

    db = KnowledgeDB()
    updated = 0
    for item in data:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict item: %s", item)
            continue
        cid = item.get("chunk_db_id")
        summ = item.get("summary")
        kws = item.get("keywords")
        if cid is None:
            logger.warning("Missing chunk_db_id, skipping: %s", item)
            continue
        if summ is not None:
            db.update_chunk_summary(cid, summ)
        if kws is not None:
            if isinstance(kws, list):
                kws = ",".join(str(k) for k in kws)
            db.update_chunk_keywords(cid, str(kws))
        db.update_chunk_status(cid, "done")
        updated += 1
    return updated


def _cleanup_orphan_files(out_dir: Path, batch_map: list) -> int:
    """Remove stale .txt/.json files that do not match the current batch_map."""
    cleaned = 0
    valid_batch_ids = {i + 1 for i in range(len(batch_map))}
    batch_id_to_set = {i + 1: set(ids) for i, ids in enumerate(batch_map)}

    for fpath in list(out_dir.glob("batch_*.txt")) + list(out_dir.glob("batch_*.json")):
        m = re.match(r"batch_(\d{3})\.(txt|json)$", fpath.name)
        if not m:
            continue
        idx = int(m.group(1))
        ext = m.group(2)

        if idx not in valid_batch_ids:
            fpath.unlink(missing_ok=True)
            cleaned += 1
            continue

        if ext == "json" and fpath.exists():
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    json_ids = {item.get("chunk_db_id") for item in data if isinstance(item, dict) and item.get("chunk_db_id") is not None}
                    if not json_ids.issubset(batch_id_to_set[idx]):
                        fpath.unlink(missing_ok=True)
                        cleaned += 1
                        # Also delete the paired txt so it gets rewritten cleanly
                        txt_path = fpath.with_suffix(".txt")
                        if txt_path.exists():
                            txt_path.unlink(missing_ok=True)
                            cleaned += 1
            except Exception:
                # If JSON is unreadable, treat as orphan
                fpath.unlink(missing_ok=True)
                cleaned += 1

    return cleaned


def run_auto_summarize(
    doc_id: str,
    out_dir: Path,
    batch_size: int = 10,
    target_chars: int = 25000,
    do_filter: bool = False,
):
    """
    Shared core for the auto-summarize pipeline.

    Returns a dict:
      - state: "pending" | "fully_summarized"
      - doc_id: str
      - applied_batches: int
      - total_batches: int
      - next_batch_txt: str   (only when state="pending")
      - next_batch_json: str  (only when state="pending")
      - pending_batches: list[dict]  (only when state="pending")
      - message: str
      - cleaned_count: int
    """
    db = KnowledgeDB()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.json"

    # Step 1: filter
    if do_filter:
        rows = db.get_embedded_but_unsummarized_chunks(doc_id)
        skipped = 0
        for row in rows:
            db_id, _, _, content, ctype, _, _, ch_title = row
            if _is_low_value(content, ch_title, chunk_type=ctype):
                db.update_chunk_status(db_id, "skipped")
                skipped += 1
        logger.info("Auto filtered low-value chunks | doc_id=%s | skipped=%s", doc_id, skipped)

    # Step 2: get pending rows
    rows = db.get_embedded_but_unsummarized_chunks(doc_id)
    current_embedded_ids = {r[0] for r in rows}

    # Step 3: load checkpoint and decide batch_map
    checkpoint = {}
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            checkpoint = {}

    batch_map = None
    done_ids = set()
    if checkpoint.get("doc_id") == doc_id:
        cp_batch_map = checkpoint.get("batch_map")
        if cp_batch_map and isinstance(cp_batch_map, list):
            cp_all_ids = set()
            for batch_ids in cp_batch_map:
                cp_all_ids.update(batch_ids)
            # If current embedded ids are a subset of the checkpoint's known ids,
            # the batch_map is still valid (some may have been done/skipped).
            if current_embedded_ids.issubset(cp_all_ids):
                batch_map = cp_batch_map
                done_ids = set(checkpoint.get("done_chunk_ids", []))
                logger.info("Resumed from checkpoint | doc_id=%s | batches=%s | done=%s", doc_id, len(batch_map), len(done_ids))

    if batch_map is None:
        # Generate fresh batch_map from current embedded rows
        batches = _smart_batch(rows, target_chars=target_chars, max_chunks=batch_size, min_chunks=3)
        batch_map = []
        for batch in batches:
            batch_map.append([row[0] for row in batch])
        done_ids = set()
        logger.info("Created new batch map | doc_id=%s | batches=%s | chunks=%s", doc_id, len(batch_map), sum(len(b) for b in batch_map))

    total_batches = len(batch_map)

    # Build a lookup from chunk_db_id -> full row
    row_by_id = {r[0]: r for r in rows}

    # Step 4: cleanup orphan files
    cleaned = _cleanup_orphan_files(out_dir, batch_map)
    if cleaned:
        logger.info("Cleaned up orphan intermediate files | count=%s", cleaned)

    # Step 5: write txt files for all batches that still have pending chunks
    for i, batch_ids in enumerate(batch_map, start=1):
        pending_ids = [bid for bid in batch_ids if bid in row_by_id and bid not in done_ids]
        if not pending_ids:
            continue
        batch = [row_by_id[bid] for bid in pending_ids]
        txt_path = out_dir / f"batch_{i:03d}.txt"
        enriched = _enrich_batch_with_images(batch, db)
        _write_batch_txt(txt_path, enriched)

    # Step 6: apply existing JSONs sequentially
    applied_count = 0
    next_to_process = total_batches + 1
    for i, batch_ids in enumerate(batch_map, start=1):
        json_path = out_dir / f"batch_{i:03d}.json"
        batch_id_set = set(batch_ids)

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Auto failed to read %s: %s. Stopping.", json_path.name, e)
                next_to_process = i
                break

            if not isinstance(data, list):
                logger.warning("Auto %s is not a JSON list. Stopping.", json_path.name)
                next_to_process = i
                break

            json_ids = {item.get("chunk_db_id") for item in data if isinstance(item, dict) and item.get("chunk_db_id") is not None}

            if not json_ids.issubset(batch_id_set):
                stale = sorted(json_ids - batch_id_set)
                logger.warning("Auto stale chunk ids in %s | batch=%s | stale=%s", json_path.name, i, stale)
                # Since _cleanup_orphan_files should have caught this, this is a safety net.
                # We clean it now and stop.
                json_path.unlink(missing_ok=True)
                next_to_process = i
                break

            count = _apply_json_file(json_path)
            applied_count += 1
            done_ids.update(json_ids)
            logger.info("Auto applied %s | chunks=%s", json_path.name, count)
            try:
                json_path.unlink()
            except Exception:
                pass

            remaining_in_batch = batch_id_set - done_ids
            if remaining_in_batch:
                logger.warning("Auto batch incomplete | batch=%s | remaining=%s", i, sorted(remaining_in_batch))
                next_to_process = i
                break
        else:
            next_to_process = i
            break

    # Step 7: remaining work or fully summarized
    if next_to_process <= total_batches:
        pending_files = [f"batch_{j:03d}.txt" for j in range(next_to_process, total_batches + 1)]
        checkpoint = {
            "doc_id": doc_id,
            "batch_map": batch_map,
            "done_chunk_ids": sorted(done_ids),
            "total_batches": total_batches,
            "applied_batches": applied_count,
            "pending_batches": pending_files,
        }
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        txt_path = out_dir / f"batch_{next_to_process:03d}.txt"
        json_path = out_dir / f"batch_{next_to_process:03d}.json"
        pending_batches = []
        for j in range(next_to_process, total_batches + 1):
            pending_batches.append({
                "txt": str(out_dir / f"batch_{j:03d}.txt"),
                "json": str(out_dir / f"batch_{j:03d}.json"),
            })
        logger.info("Auto progress | applied=%s/%s | next_batch=%s", applied_count, total_batches, txt_path.name)
        return {
            "state": "pending",
            "doc_id": doc_id,
            "applied_batches": applied_count,
            "total_batches": total_batches,
            "next_batch_txt": str(txt_path),
            "next_batch_json": str(json_path),
            "pending_batches": pending_batches,
            "message": f"Progress: {applied_count}/{total_batches} batch(es) applied. Next batch: {txt_path.name}",
            "cleaned_count": cleaned,
        }

    # All done
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    cleaned_end = 0
    for txt_path in out_dir.glob("batch_*.txt"):
        txt_path.unlink()
        cleaned_end += 1
    if cleaned_end:
        logger.info("Auto cleaned up intermediate files | count=%s", cleaned_end)
    return {
        "state": "fully_summarized",
        "doc_id": doc_id,
        "applied_batches": applied_count,
        "total_batches": total_batches,
        "message": f"All {total_batches} batch(es) applied. Document {doc_id} is fully summarized.",
        "cleaned_count": cleaned + cleaned_end,
    }


def cmd_list(args):
    db = KnowledgeDB()
    rows = db.get_embedded_but_unsummarized_chunks(args.doc_id)
    total = len(rows)
    print(f"Total pending chunks for doc {args.doc_id}: {total}")
    if total == 0:
        return
    for db_id, doc_id, chunk_id, content, ctype, ps, pe, ch_title in rows:
        preview = content[:120].replace("\n", " ")
        print(f"  chunk_db_id={db_id} | {doc_id} | {chunk_id} | {ch_title} P{ps}-{pe} | {preview}...")


def cmd_dump(args):
    db = KnowledgeDB()
    rows = db.get_embedded_but_unsummarized_chunks(args.doc_id)
    total = len(rows)
    start = max(0, args.offset)
    end = start + args.batch_size
    batch = rows[start:end]
    print(f"Dumping {len(batch)} chunk(s) (offset {start}, batch size {args.batch_size}, total pending {total})")
    if not batch:
        return

    enriched = _enrich_batch_with_images(batch, db)
    _write_batch_txt(Path(args.output), enriched)
    print(f"Written to {Path(args.output).resolve()}")


def cmd_apply(args):
    in_path = Path(args.input)
    if not _validate_inside_skill(in_path):
        skill_root = Config.DB_PATH.parent.parent
        logger.error("Input file outside skill directory | skill_root=%s", skill_root)
        sys.exit(1)

    updated = _apply_json_file(in_path)
    logger.info("Total updated: %s", updated)


def cmd_filter(args):
    db = KnowledgeDB()
    rows = db.get_embedded_but_unsummarized_chunks(args.doc_id)
    skipped = 0
    for db_id, doc_id, chunk_id, content, ctype, ps, pe, ch_title in rows:
        if _is_low_value(content, ch_title, chunk_type=ctype):
            db.update_chunk_status(db_id, "skipped")
            skipped += 1
    logger.info("Filtered low-value chunks | doc_id=%s | skipped=%s", args.doc_id, skipped)


def cmd_progress(args):
    db = KnowledgeDB()
    with db._connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ?", (args.doc_id,)
        ).fetchone()["c"]
        done = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ? AND status = 'done'", (args.doc_id,)
        ).fetchone()["c"]
        embedded = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ? AND status = 'embedded'", (args.doc_id,)
        ).fetchone()["c"]
        skipped = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ? AND status = 'skipped'", (args.doc_id,)
        ).fetchone()["c"]
        pending = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ? AND status = 'pending'", (args.doc_id,)
        ).fetchone()["c"]
        failed = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE doc_id = ? AND status = 'failed'", (args.doc_id,)
        ).fetchone()["c"]

    def bar(count, total, width=20):
        if total == 0:
            return " " * width
        filled = int(width * count / total)
        return "█" * filled + "░" * (width - filled)

    print(f"Doc: {args.doc_id}")
    print(f"  total   : {total:>4}")
    print(f"  done    : {done:>4} {bar(done, total)}")
    print(f"  embedded: {embedded:>4} {bar(embedded, total)}")
    print(f"  skipped : {skipped:>4} {bar(skipped, total)}")
    print(f"  pending : {pending:>4} {bar(pending, total)}")
    print(f"  failed  : {failed:>4} {bar(failed, total)}")


def _resolve_output_dir(base_dir: str, doc_id: str) -> Path:
    """Append doc_id as subdirectory if the provided path does not already end with it."""
    p = Path(base_dir)
    # If the last path component is already the doc_id, use as-is
    if p.name == doc_id:
        return p
    return p / doc_id


def cmd_auto(args):
    out_dir = _resolve_output_dir(args.output_dir, args.doc_id)
    result = run_auto_summarize(
        doc_id=args.doc_id,
        out_dir=out_dir,
        batch_size=args.batch_size,
        target_chars=args.target_chars,
        do_filter=args.filter,
    )

    if result["state"] == "pending":
        print(f"\nNext batch to summarize: {Path(result['next_batch_txt']).name}")
        print(f"  1. Have Agent read '{Path(result['next_batch_txt']).resolve()}'")
        print(f"  2. Write result to '{Path(result['next_batch_json']).resolve()}' (JSON array)")
        print(f"  3. Rerun: python agent_batch_helper.py auto --doc-id {args.doc_id} --output-dir {args.output_dir}")
        return

    print(result["message"])


def main():
    Config.setup_logging()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="Agent batch helper for work-docs-library")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List all pending chunks for a doc")
    p_list.add_argument("--doc-id", required=True)
    p_list.set_defaults(func=cmd_list)

    p_dump = sub.add_parser("dump", help="Dump a batch of pending chunks to a text file for Agent review")
    p_dump.add_argument("--doc-id", required=True)
    p_dump.add_argument("--batch-size", type=int, default=20)
    p_dump.add_argument("--offset", type=int, default=0)
    p_dump.add_argument("--output", required=True, help="Path to output text file")
    p_dump.set_defaults(func=cmd_dump)

    p_apply = sub.add_parser("apply", help="Apply batch summaries/keywords from a JSON file")
    p_apply.add_argument("--input", required=True, help="Path to JSON file: [{chunk_db_id, summary, keywords}, ...]")
    p_apply.set_defaults(func=cmd_apply)

    p_filter = sub.add_parser("filter", help="Auto-mark low-value chunks (packaging, disclaimers, TOC) as skipped")
    p_filter.add_argument("--doc-id", required=True)
    p_filter.set_defaults(func=cmd_filter)

    p_progress = sub.add_parser("progress", help="Show summarization progress for a document")
    p_progress.add_argument("--doc-id", required=True)
    p_progress.set_defaults(func=cmd_progress)

    p_auto = sub.add_parser("auto", help="Auto pipeline: filter, smart-batch, dump, and coordinate apply with checkpoint/resume")
    p_auto.add_argument("--doc-id", required=True)
    p_auto.add_argument("--output-dir", default="./auto_batches", help="Directory for batch .txt/.json files and checkpoint")
    p_auto.add_argument("--batch-size", type=int, default=10, help="Max chunks per batch")
    p_auto.add_argument("--target-chars", type=int, default=25000, help="Target characters per batch")
    p_auto.add_argument("--filter", action="store_true", help="First filter low-value chunks")
    p_auto.add_argument("--parallel", type=int, default=1, help="Reserved for future concurrent Agent support")
    p_auto.set_defaults(func=cmd_auto)

    args = parser.parse_args()
    args.func(args)
    sys.exit(0)


if __name__ == "__main__":
    main()
