import json
from pathlib import Path

import pytest

from core.config import Config
from core.db import KnowledgeDB
from core.models import Chunk, Document, Chapter
import agent_batch_helper


@pytest.fixture
def patched_config(monkeypatch, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    monkeypatch.setattr(Config, "DB_PATH", kb / "workdocs.db")
    monkeypatch.setattr(Config, "FAISS_INDEX_PATH", kb / "faiss.index")
    monkeypatch.setattr(Config, "ID_MAP_PATH", kb / "id_map.json")
    return tmp_path


@pytest.fixture
def sample_doc_with_chunks(patched_config):
    db = KnowledgeDB()
    doc = Document(
        doc_id="d1", title="Test", source_path=str(patched_config / "a.pdf"), file_type="pdf",
        total_pages=2, chapters=[Chapter(title="Ch1", start_page=1, end_page=2, level=1)]
    )
    db.upsert_document(doc)
    for i in range(3):
        cid = db.insert_chunk(Chunk(
            doc_id="d1", chunk_id=f"p{i+1}", content=f"content {i+1}",
            chunk_type="text", page_start=i+1, page_end=i+1
        ))
        db.update_chunk_status(cid, "embedded")
    return "d1"


def test_cmd_list(patched_config, sample_doc_with_chunks, capsys):
    args = type("Args", (), {"doc_id": sample_doc_with_chunks})()
    agent_batch_helper.cmd_list(args)
    captured = capsys.readouterr()
    assert "Total pending chunks" in captured.out
    assert "chunk_db_id=" in captured.out


def test_cmd_dump(patched_config, sample_doc_with_chunks, capsys):
    out = patched_config / "dump.txt"
    args = type("Args", (), {
        "doc_id": sample_doc_with_chunks,
        "batch_size": 2,
        "offset": 0,
        "output": str(out)
    })()
    agent_batch_helper.cmd_dump(args)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "CHUNK_DB_ID=" in text
    assert "content 1" in text


def test_cmd_apply(patched_config, sample_doc_with_chunks, capsys):
    db = KnowledgeDB()
    rows = db.get_embedded_but_unsummarized_chunks("d1")
    first_id = rows[0][0]

    json_path = patched_config / "apply.json"
    json_path.write_text(json.dumps([
        {"chunk_db_id": first_id, "summary": "Applied summary", "keywords": "x,y"}
    ]), encoding="utf-8")

    args = type("Args", (), {"input": str(json_path)})()
    agent_batch_helper.cmd_apply(args)

    ck = db.get_chunk_by_db_id(first_id)
    assert ck.summary == "Applied summary"
    assert ck.keywords == ["x", "y"]
    assert ck.status == "done"


def test_cmd_apply_path_outside_skill_rejected(patched_config, sample_doc_with_chunks, capsys):
    import tempfile
    json_path = Path(tempfile.gettempdir()) / "outside.json"
    json_path.write_text("[]", encoding="utf-8")
    args = type("Args", (), {"input": str(json_path)})()
    with pytest.raises(SystemExit):
        agent_batch_helper.cmd_apply(args)
    json_path.unlink()


def test_cmd_apply_keywords_as_list(patched_config, sample_doc_with_chunks, capsys):
    db = KnowledgeDB()
    rows = db.get_embedded_but_unsummarized_chunks("d1")
    first_id = rows[0][0]

    json_path = patched_config / "apply2.json"
    json_path.write_text(json.dumps([
        {"chunk_db_id": first_id, "summary": "S2", "keywords": ["a", "b"]}
    ]), encoding="utf-8")

    args = type("Args", (), {"input": str(json_path)})()
    agent_batch_helper.cmd_apply(args)

    ck = db.get_chunk_by_db_id(first_id)
    assert ck.keywords == ["a", "b"]
    assert ck.status == "done"


def test_cmd_filter(patched_config, sample_doc_with_chunks, capsys):
    db = KnowledgeDB()
    # Insert a low-value chunk
    low_cid = db.insert_chunk(Chunk(
        doc_id="d1", chunk_id="p4", content="IMPORTANT NOTICE AND DISCLAIMER",
        chunk_type="text", page_start=4, page_end=4, chapter_title="Legal"
    ))
    db.update_chunk_status(low_cid, "embedded")

    args = type("Args", (), {"doc_id": "d1"})()
    agent_batch_helper.cmd_filter(args)

    ck_low = db.get_chunk_by_db_id(low_cid)
    assert ck_low.status == "skipped"


def test_cmd_progress(patched_config, sample_doc_with_chunks, capsys):
    db = KnowledgeDB()
    # Mark one chunk done and one pending (create an extra chunk as pending)
    rows = db.get_embedded_but_unsummarized_chunks("d1")
    db.update_chunk_status(rows[0][0], "done")

    pend_cid = db.insert_chunk(Chunk(
        doc_id="d1", chunk_id="p0", content="pending chunk",
        chunk_type="text", page_start=0, page_end=0
    ))

    skip_cid = db.insert_chunk(Chunk(
        doc_id="d1", chunk_id="p9", content="skip me",
        chunk_type="text", page_start=9, page_end=9
    ))
    db.update_chunk_status(skip_cid, "skipped")

    args = type("Args", (), {"doc_id": "d1"})()
    agent_batch_helper.cmd_progress(args)
    captured = capsys.readouterr()
    assert "total" in captured.out
    assert "done" in captured.out
    assert "embedded" in captured.out


def test_cmd_auto_creates_batches_and_checkpoint(patched_config, sample_doc_with_chunks, capsys):
    out_dir = patched_config / "auto_batches"
    args = type("Args", (), {
        "doc_id": "d1",
        "output_dir": str(out_dir),
        "batch_size": 2,
        "target_chars": 25000,
        "filter": False,
        "parallel": 1,
    })()
    agent_batch_helper.cmd_auto(args)
    captured = capsys.readouterr()

    # Should create 2 batches (3 chunks, batch_size=2 -> batch_001 with 2, batch_002 with 1 merged into 001 because orphan)
    # Actually smart_batch merges orphan < min_chunks=3 into previous, so 1 batch of 3
    assert (out_dir / "batch_001.txt").exists()
    # Checkpoint should NOT exist because there is no batch_001.json to apply, so it stops at batch 1
    assert (out_dir / "checkpoint.json").exists()
    checkpoint = json.loads((out_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["doc_id"] == "d1"
    assert checkpoint["applied_batches"] == 0
    assert "batch_001.txt" in checkpoint["pending_batches"]


def test_cmd_auto_applies_existing_json_and_resumes(patched_config, sample_doc_with_chunks, capsys, caplog):
    db = KnowledgeDB()
    rows = db.get_embedded_but_unsummarized_chunks("d1")
    out_dir = patched_config / "auto_batches"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create batch txt manually to bypass auto dump size logic
    # We create a single-batch scenario by making batch_001.txt and batch_001.json
    batch_txt = out_dir / "batch_001.txt"
    batch_json = out_dir / "batch_002.json"
    batch_txt.write_text("dummy", encoding="utf-8")

    # Write a second batch JSON that should be applied
    batch_json.write_text(json.dumps([
        {"chunk_db_id": rows[0][0], "summary": "Auto summary", "keywords": "kw1,kw2"}
    ]), encoding="utf-8")

    # However auto will regenerate batches based on current DB rows.
    # Let's just let auto run normally with small batch size
    args = type("Args", (), {
        "doc_id": "d1",
        "output_dir": str(out_dir),
        "batch_size": 10,
        "target_chars": 25000,
        "filter": False,
        "parallel": 1,
    })()
    agent_batch_helper.cmd_auto(args)
    captured = capsys.readouterr()

    # Since no JSON exists for batch_001, it should stop there.
    assert "Progress: 0/1 batch(es) applied" in captured.out or "batch_001.txt" in captured.out

    # Now create JSON for batch_001 and rerun
    (out_dir / "batch_001.json").write_text(json.dumps([
        {"chunk_db_id": rows[0][0], "summary": "S1", "keywords": "a"},
        {"chunk_db_id": rows[1][0], "summary": "S2", "keywords": "b"},
        {"chunk_db_id": rows[2][0], "summary": "S3", "keywords": "c"},
    ]), encoding="utf-8")

    agent_batch_helper.cmd_auto(args)
    captured2 = capsys.readouterr()

    assert "All 1 batch(es) applied" in captured2.out or "Auto complete" in caplog.text
    ck = db.get_chunk_by_db_id(rows[0][0])
    assert ck.status == "done"
    assert ck.summary == "S1"
    assert not (out_dir / "checkpoint.json").exists()


def test_cmd_auto_with_filter(patched_config, sample_doc_with_chunks, capsys, caplog):
    db = KnowledgeDB()
    low_cid = db.insert_chunk(Chunk(
        doc_id="d1", chunk_id="p4", content="DISCLAIMER: do not use in safety critical applications",
        chunk_type="text", page_start=4, page_end=4, chapter_title="Legal"
    ))
    db.update_chunk_status(low_cid, "embedded")

    out_dir = patched_config / "auto_batches"
    args = type("Args", (), {
        "doc_id": "d1",
        "output_dir": str(out_dir),
        "batch_size": 10,
        "target_chars": 25000,
        "filter": True,
        "parallel": 1,
    })()
    agent_batch_helper.cmd_auto(args)
    captured = capsys.readouterr()
    assert "low-value chunk(s) as skipped" in captured.out or "Auto filtered low-value chunks" in caplog.text
    ck = db.get_chunk_by_db_id(low_cid)
    assert ck.status == "skipped"


def test_cmd_auto_detects_incomplete_batch_after_partial_json(patched_config, capsys, caplog):
    db = KnowledgeDB()
    doc = Document(
        doc_id="d2", title="Test", source_path=str(patched_config / "b.pdf"), file_type="pdf",
        total_pages=2, chapters=[Chapter(title="Ch1", start_page=1, end_page=2, level=1)]
    )
    db.upsert_document(doc)
    cids = []
    for i in range(4):
        cid = db.insert_chunk(Chunk(
            doc_id="d2", chunk_id=f"p{i+1}", content=f"content {i+1}",
            chunk_type="text", page_start=i+1, page_end=i+1
        ))
        db.update_chunk_status(cid, "embedded")
        cids.append(cid)

    out_dir = patched_config / "auto_batches"
    args = type("Args", (), {
        "doc_id": "d2",
        "output_dir": str(out_dir),
        "batch_size": 10,
        "target_chars": 25000,
        "filter": False,
        "parallel": 1,
    })()

    # First run generates a single batch of 4 chunks
    agent_batch_helper.cmd_auto(args)
    captured = capsys.readouterr()
    assert "batch_001.txt" in captured.out

    # Simulate a partial JSON that covers only 2 of the 4 chunks
    partial_json = out_dir / "batch_001.json"
    partial_data = [{"chunk_db_id": cids[i], "summary": "S", "keywords": "k"} for i in range(2)]
    partial_json.write_text(json.dumps(partial_data), encoding="utf-8")

    # Re-run auto
    agent_batch_helper.cmd_auto(args)
    captured = capsys.readouterr()

    # Should detect incomplete batch and warn instead of declaring done
    assert "Batch 1 still has 2 unprocessed chunk(s)" in captured.out or "Auto batch incomplete" in caplog.text
    assert "Please remove batch_001.json and rerun" in captured.out or "Auto JSON may be stale or incomplete" in caplog.text

    # Checkpoint should be written with only the 2 applied ids
    checkpoint = json.loads((out_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert set(checkpoint["done_chunk_ids"]) == set(cids[:2])


def test_content_keywords_prefix_ratio_avoids_footer_false_positive(patched_config, capsys):
    db = KnowledgeDB()
    # Content with copyright only at the end (simulating a footer)
    long_content = "A" * 2000 + "\nCopyright 2025 Example Corp"
    cid = db.insert_chunk(Chunk(
        doc_id="d1", chunk_id="p10", content=long_content,
        chunk_type="text", page_start=10, page_end=10, chapter_title="Technical Section"
    ))
    db.update_chunk_status(cid, "embedded")

    args = type("Args", (), {"doc_id": "d1"})()
    agent_batch_helper.cmd_filter(args)
    ck = db.get_chunk_by_db_id(cid)
    # With prefix ratio 0.35, copyright at index ~2000 is outside the prefix window
    assert ck.status == "embedded"

    # Now insert a pure disclaimer page where the keyword is at the top
    short_disclaimer = "IMPORTANT NOTICE AND DISCLAIMER\nDo not use..."
    cid2 = db.insert_chunk(Chunk(
        doc_id="d1", chunk_id="p11", content=short_disclaimer,
        chunk_type="text", page_start=11, page_end=11, chapter_title="Legal"
    ))
    db.update_chunk_status(cid2, "embedded")
    agent_batch_helper.cmd_filter(args)
    ck2 = db.get_chunk_by_db_id(cid2)
    assert ck2.status == "skipped"


def test_cmd_auto_detects_stale_json_after_batch_reorg(patched_config, capsys, caplog):
    db = KnowledgeDB()
    doc = Document(
        doc_id="d3", title="Test", source_path=str(patched_config / "c.pdf"), file_type="pdf",
        total_pages=2, chapters=[Chapter(title="Ch1", start_page=1, end_page=2, level=1)]
    )
    db.upsert_document(doc)
    cids = []
    for i in range(5):
        cid = db.insert_chunk(Chunk(
            doc_id="d3", chunk_id=f"p{i+1}", content=f"content {i+1}",
            chunk_type="text", page_start=i+1, page_end=i+1
        ))
        db.update_chunk_status(cid, "embedded")
        cids.append(cid)

    out_dir = patched_config / "auto_batches_d3"
    args = type("Args", (), {
        "doc_id": "d3",
        "output_dir": str(out_dir),
        "batch_size": 3,
        "target_chars": 25000,
        "filter": False,
        "parallel": 1,
    })()

    # First run: 5 chunks with batch_size=3 -> [3,2] then merge orphan -> [5]
    agent_batch_helper.cmd_auto(args)
    captured = capsys.readouterr()
    assert "batch_001.txt" in captured.out

    # Manually apply first 3 chunks so they become done
    for cid in cids[:3]:
        db.update_chunk_summary(cid, "summary")
        db.update_chunk_keywords(cid, "k")
        db.update_chunk_status(cid, "done")

    # Re-run auto: pending=2 -> [2]
    agent_batch_helper.cmd_auto(args)
    captured = capsys.readouterr()
    assert "batch_001.txt" in captured.out

    # Now place a stale JSON from the old [3,2] grouping (covers 3 chunks)
    stale_json = out_dir / "batch_001.json"
    stale_data = [{"chunk_db_id": cids[i], "summary": "S", "keywords": "k"} for i in range(3)]
    stale_json.write_text(json.dumps(stale_data), encoding="utf-8")

    # Re-run auto again
    agent_batch_helper.cmd_auto(args)
    captured = capsys.readouterr()

    # Should detect stale JSON because its ids are not a subset of the current batch
    assert "contains chunk ids not in current batch 1" in captured.out or "Auto stale chunk ids" in caplog.text
    assert "Please remove stale JSON files and rerun" in captured.out or "Auto please remove stale JSON files" in caplog.text
