"""Evaluation persistence tests."""

import pytest
from core.db import KnowledgeDB
from core.evaluation import EvalDataset, EvalQuestion


def test_eval_dataset_crud(tmp_path, monkeypatch):
    db = KnowledgeDB()
    q = EvalQuestion(
        question="What is the reset sequence of SPI?",
        ground_truth_answer="CS low, clock idle, then data.",
        ground_truth_context_ids=[1, 2],
        ground_truth_doc_ids=["doc-a"],
        tags=["SPI", "reset"],
    )
    ds = EvalDataset(name="baseline_v1", questions=[q])
    db.save_eval_dataset(ds)
    loaded = db.load_eval_dataset("baseline_v1")
    assert len(loaded.questions) == 1
    assert loaded.questions[0].question == q.question


def test_list_and_delete_eval_datasets(tmp_path, monkeypatch):
    db = KnowledgeDB()
    db.save_eval_dataset(EvalDataset(name="ds1", questions=[]))
    db.save_eval_dataset(EvalDataset(name="ds2", questions=[]))
    names = db.list_eval_datasets()
    assert "ds1" in names
    assert "ds2" in names
    assert db.delete_eval_dataset("ds1")
    assert "ds1" not in db.list_eval_datasets()
    assert not db.delete_eval_dataset("nonexistent")


def test_load_missing_eval_dataset_raises(tmp_path, monkeypatch):
    db = KnowledgeDB()
    with pytest.raises(ValueError):
        db.load_eval_dataset("missing")
