"""Checks on the evaluation corpus and the scoring. These use mock models only; they test the harness."""
import json
import os

import pytest

from core.pdf import extract_pdf
from eval.corpus import REAL, SYNTHETIC, load_document
from eval.metrics import extraction_check
from eval.run import QUESTIONS, run

with open(QUESTIONS) as f:
    LABELS = json.load(f)
QUESTION_LIST = LABELS["questions"]
REAL_CACHED = all(os.path.exists(f".eval-cache/{doc}.pdf") for doc in REAL)


def test_every_question_has_known_provenance_and_valid_fields():
    for q in QUESTION_LIST:
        assert q["label_source"] in LABELS["label_sources"], q["id"]
        assert q["doc"] in SYNTHETIC or q["doc"] in REAL, q["id"]
        if q["type"] == "unanswerable":
            assert "evidence" not in q and "answer_patterns" not in q, q["id"]
        else:
            assert q["evidence"] and q["answer_patterns"], q["id"]


def test_corpus_covers_required_question_kinds():
    kinds = {q["type"] for q in QUESTION_LIST}
    assert {"answerable", "table", "unanswerable"} <= kinds
    assert any("expected_omission" in q for q in QUESTION_LIST)


def _check_spans(doc_ids):
    for doc_id in doc_ids:
        result = extract_pdf(load_document(doc_id))
        for q in (q for q in QUESTION_LIST if q["doc"] == doc_id):
            for check in extraction_check(q, result["pages"], result["report"]):
                # A labelled span must be extracted unless the label says extraction is known to lose it
                assert check["extracted"] != ("expected_omission" in q), q["id"]


def test_synthetic_spans_match_extraction():
    _check_spans(SYNTHETIC)


@pytest.mark.skipif(not REAL_CACHED, reason="real papers not cached; run `python -m eval.run --docs real` once")
def test_real_spans_match_extraction():
    _check_spans(REAL)


def test_scanned_page_omission_is_flagged():
    result = extract_pdf(load_document("syn-scanned"))
    q = next(q for q in QUESTION_LIST if q["id"] == "syn-s1")
    assert extraction_check(q, result["pages"], result["report"]) == [
        {"page": 2, "extracted": False, "flagged_if_missing": True}]


def test_oracle_scores_are_perfect_except_for_extraction_omissions():
    report = run("mock-oracle", list(SYNTHETIC), ["baseline", "evidence"])
    evidence = report["modes"]["evidence"]
    assert evidence["summary"]["answer_accuracy"] == 1.0
    assert evidence["summary"]["abstention_recall"] == 1.0
    unsupported = [s["id"] for s in evidence["per_question"] if s["type"] != "unanswerable" and not s["supported"]]
    assert unsupported == ["syn-s1"]  # the scanned page: nothing to quote from
    assert report["modes"]["baseline"]["summary"]["citation_precision"] is None


def test_flawed_mock_errors_are_detected():
    report = run("mock-flawed", list(SYNTHETIC), ["evidence"])
    per = {s["id"]: s for s in report["modes"]["evidence"]["per_question"]}
    assert any(s["quotes_wrong_page"] for s in per.values())
    assert any(s["quotes_not_found"] for s in per.values())
    assert any(s["type"] == "unanswerable" and not s["abstained"] for s in per.values())
    assert any(s["type"] != "unanswerable" and not s["correct"] for s in per.values())
