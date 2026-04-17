#!/usr/bin/env python3
"""
Auto-read batch txt files and generate batch JSON using the configured LLM.
Then apply them via run_auto_summarize loop.
"""
import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_SKILL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SKILL_ROOT))

from core.config import Config
from core.llm_client import ChatClient
import agent_batch_helper as abh

PROMPT_TEXT = (Config.PROMPT_DIR / "summarize.txt").read_text(encoding="utf-8")


def _parse_batch_txt(txt_path: Path):
    """Parse batch txt into list of (chunk_db_id, content)."""
    text = txt_path.read_text(encoding="utf-8")
    chunks = []
    pattern = re.compile(r"--- CHUNK_DB_ID=(\d+)\s*\|.*?---\n(.*?)(?=\n--- CHUNK_DB_ID=|\n={80}\n*\Z)", re.DOTALL)
    for m in pattern.finditer(text):
        cid = int(m.group(1))
        content = m.group(2).strip()
        chunks.append((cid, content))
    return chunks


def _summarize_chunk(client: ChatClient, cid: int, content: str):
    prompt = PROMPT_TEXT.replace("{{text}}", content[:12000])
    try:
        raw = client.chat([{"role": "user", "content": prompt}], temperature=0.3)
    except Exception as e:
        return (cid, None, str(e))
    summary = ""
    keywords = ""
    for line in raw.splitlines():
        if line.lower().startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
        elif line.lower().startswith("keywords:"):
            keywords = line.split(":", 1)[1].strip()
    if not summary:
        summary = raw.strip()
    return (cid, summary, keywords)


def _generate_json_for_batch(client: ChatClient, txt_path: Path, json_path: Path, max_workers: int = 3):
    chunks = _parse_batch_txt(txt_path)
    if not chunks:
        return 0
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_summarize_chunk, client, cid, content): cid for cid, content in chunks}
        for fut in as_completed(futures):
            cid, summary, keywords = fut.result()
            if summary is not None:
                results.append({"chunk_db_id": cid, "summary": summary, "keywords": keywords})
            else:
                print(f"  WARN chunk {cid} failed: {keywords}", file=sys.stderr)
    results.sort(key=lambda x: [c[0] for c in chunks].index(x["chunk_db_id"]))
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(results)


def run_for_doc(doc_id: str, out_dir_base: str = "./auto_batches", max_workers: int = 3):
    out_dir = abh._resolve_output_dir(out_dir_base, doc_id)
    client = ChatClient()
    round_num = 0
    while True:
        round_num += 1
        result = abh.run_auto_summarize(
            doc_id=doc_id,
            out_dir=out_dir,
            batch_size=10,
            target_chars=25000,
            do_filter=False,
        )
        state = result["state"]
        applied = result["applied_batches"]
        total = result["total_batches"]
        print(f"[Round {round_num}] state={state} applied={applied}/{total}")
        if state == "fully_summarized":
            print(f"Done: {result['message']}")
            break

        pending = result.get("pending_batches", [])
        print(f"  Processing {len(pending)} pending batch(es)...")
        for item in pending:
            txt_path = Path(item["txt"])
            json_path = Path(item["json"])
            if not txt_path.exists():
                continue
            print(f"  -> {txt_path.name}")
            count = _generate_json_for_batch(client, txt_path, json_path, max_workers=max_workers)
            print(f"     generated {count} summaries")
            # Small sleep to avoid hammering the API
            time.sleep(0.5)

    client.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--output-dir", default="./auto_batches")
    parser.add_argument("--max-workers", type=int, default=3, help="LLM concurrency per batch")
    args = parser.parse_args()
    run_for_doc(args.doc_id, args.output_dir, args.max_workers)
