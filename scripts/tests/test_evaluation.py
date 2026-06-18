"""Evaluation persistence tests."""

import pytest
from core.db import KnowledgeDB
from core.models import EvalDataset, EvalQuestion


def test_eval_dataset_crud(tmp_path):
    db = KnowledgeDB(db_path=tmp_path / "eval.db")
    q = EvalQuestion(
        question="What is the reset sequence of SPI?",
        ground_truth_answer="CS low, clock idle, then data.",
        ground_truth_context_ids=[1, 2],
        ground_truth_doc_ids=["doc-a"],
        tags=["SPI", "reset"],
        metadata={"difficulty": "easy"},
    )
    ds = EvalDataset(name="baseline_v1", questions=[q], metadata={"version": "v1"})
    db.save_eval_dataset(ds)
    loaded = db.load_eval_dataset("baseline_v1")
    assert loaded.name == "baseline_v1"
    assert loaded.metadata == {"version": "v1"}
    assert len(loaded.questions) == 1
    loaded_q = loaded.questions[0]
    assert loaded_q.question == q.question
    assert loaded_q.ground_truth_answer == q.ground_truth_answer
    assert loaded_q.ground_truth_context_ids == q.ground_truth_context_ids
    assert loaded_q.ground_truth_doc_ids == q.ground_truth_doc_ids
    assert loaded_q.tags == q.tags
    assert loaded_q.metadata == q.metadata


def test_list_and_delete_eval_datasets(tmp_path):
    db = KnowledgeDB(db_path=tmp_path / "eval.db")
    db.save_eval_dataset(EvalDataset(name="ds1", questions=[]))
    db.save_eval_dataset(EvalDataset(name="ds2", questions=[]))
    names = db.list_eval_datasets()
    assert "ds1" in names
    assert "ds2" in names
    assert db.delete_eval_dataset("ds1")
    assert "ds1" not in db.list_eval_datasets()
    assert not db.delete_eval_dataset("nonexistent")


def test_load_missing_eval_dataset_raises(tmp_path):
    db = KnowledgeDB(db_path=tmp_path / "eval.db")
    with pytest.raises(ValueError):
        db.load_eval_dataset("missing")


def test_save_eval_dataset_replaces_existing(tmp_path):
    db = KnowledgeDB(db_path=tmp_path / "eval.db")
    old_q = EvalQuestion(
        question="Old question?",
        ground_truth_answer="Old answer.",
        ground_truth_context_ids=[1],
        ground_truth_doc_ids=["old-doc"],
        tags=["old"],
    )
    db.save_eval_dataset(EvalDataset(name="replace_ds", questions=[old_q]))

    new_q = EvalQuestion(
        question="New question?",
        ground_truth_answer="New answer.",
        ground_truth_context_ids=[2, 3],
        ground_truth_doc_ids=["new-doc"],
        tags=["new"],
    )
    db.save_eval_dataset(EvalDataset(name="replace_ds", questions=[new_q]))

    loaded = db.load_eval_dataset("replace_ds")
    assert loaded.name == "replace_ds"
    assert len(loaded.questions) == 1
    loaded_q = loaded.questions[0]
    assert loaded_q.question == new_q.question
    assert loaded_q.ground_truth_answer == new_q.ground_truth_answer
    assert loaded_q.ground_truth_context_ids == new_q.ground_truth_context_ids
    assert loaded_q.ground_truth_doc_ids == new_q.ground_truth_doc_ids
    assert loaded_q.tags == new_q.tags
