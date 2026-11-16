"""Scoring for the evaluation. Pure functions; no model or network access."""
import re

from core.evidence import is_abstention, normalize, split_answer, verify_quotes


def extraction_check(question, pages, report):
    """Whether each gold span survived extraction, and whether a missing one was flagged by the report."""
    results = []
    for ev in question.get("evidence", []):
        span = normalize(ev["span"])
        on_page = ev["page"] <= len(pages) and span in normalize(pages[ev["page"] - 1])
        flagged = (ev["page"] in report["pages_without_text"] or report["repaired"]
                   or ev["page"] > report["pages_processed"])
        results.append({"page": ev["page"], "extracted": on_page, "flagged_if_missing": on_page or flagged})
    return results


def score_answer(question, raw_answer, pages, full_text):
    body, cited = split_answer(raw_answer)
    checks = verify_quotes(cited, pages, full_text)
    abstained = is_abstention(body)
    gold_pages = {ev["page"] for ev in question.get("evidence", [])}
    verified = [c for c in checks if c["status"] == "verified"]
    result = {
        "id": question["id"],
        "type": question["type"],
        "abstained": abstained,
        "quotes_cited": len(checks),
        "quotes_verified": len(verified),
        "quotes_wrong_page": sum(c["status"] == "wrong_page" for c in checks),
        "quotes_not_found": sum(c["status"] == "not_found" for c in checks),
        "cites_gold_page": any(c["found_page"] in gold_pages for c in verified),
    }
    if question["type"] != "unanswerable":
        text = normalize(body)
        result["correct"] = not abstained and any(re.search(p, text) for p in question["answer_patterns"])
        result["supported"] = result["correct"] and bool(verified)
    return result


def _rate(numerator, denominator):
    return round(numerator / denominator, 3) if denominator else None


def summarize(scores):
    answerable = [s for s in scores if s["type"] != "unanswerable"]
    unanswerable = [s for s in scores if s["type"] == "unanswerable"]
    cited = sum(s["quotes_cited"] for s in scores)
    return {
        "questions": len(scores),
        "answer_accuracy": _rate(sum(s["correct"] for s in answerable), len(answerable)),
        "answer_supported": _rate(sum(s["supported"] for s in answerable), len(answerable)),
        "citation_precision": _rate(sum(s["quotes_verified"] for s in scores), cited),
        "citations_on_gold_page": _rate(sum(s["cites_gold_page"] for s in answerable), len(answerable)),
        "abstention_recall": _rate(sum(s["abstained"] for s in unanswerable), len(unanswerable)),
        "false_abstention_rate": _rate(sum(s["abstained"] for s in answerable), len(answerable)),
    }
