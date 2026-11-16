"""Run the evaluation.

    python -m eval.run                          # mock models only: checks the harness, not model quality
    python -m eval.run --docs synthetic         # no network needed
    python -m eval.run --model gemini:gemini-flash-latest --allow-api-calls   # real model, uses your API quota

Mock models answer from the gold labels (oracle) or with planted mistakes (flawed). Their scores show the
metrics behave as intended; they say nothing about how well Gemini answers questions.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from core.evidence import ABSTAIN_PHRASE, baseline_prompt, build_context, evidence_prompt
from core.pdf import extract_pdf, extraction_warnings
from eval.corpus import REAL, SYNTHETIC, load_document
from eval.metrics import extraction_check, score_answer, summarize

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "400000"))
QUESTIONS = os.path.join(os.path.dirname(__file__), "questions.json")


def oracle_model(prompt, question, mode):
    if question["type"] == "unanswerable":
        return ABSTAIN_PHRASE
    answer = f"The paper states: {question['answer']}."
    if mode == "baseline":
        return answer
    quotes = "\n".join(f'- [Page {ev["page"]}] "{ev["span"]}"' for ev in question["evidence"])
    return f"{answer}\n\nEvidence:\n{quotes}"


def flawed_model(prompt, question, mode):
    """Deterministic mistakes: wrong page, invented quote, no abstention, and a wrong answer."""
    flaw = sum(map(ord, question["id"])) % 4
    if question["type"] == "unanswerable":
        return "It cost roughly $2 million." if flaw < 2 else ABSTAIN_PHRASE
    ev = question["evidence"][0]
    if flaw == 0:
        quote = f'- [Page {ev["page"] + 1}] "{ev["span"]}"'
    elif flaw == 1:
        quote = f'- [Page {ev["page"]}] "this sentence does not appear anywhere in the paper"'
    elif flaw == 2:
        wrong = "The paper reports a value of 999."
        return wrong if mode == "baseline" else f'{wrong}\n\nEvidence:\n- [Page {ev["page"]}] "{ev["span"]}"'
    else:
        quote = f'- [Page {ev["page"]}] "{ev["span"]}"'
    answer = f"The paper states: {question['answer']}."
    return answer if mode == "baseline" else f"{answer}\n\nEvidence:\n{quote}"


def gemini_model(model_name):
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GOOGLE_API_KEY to evaluate a real model.")
    client = genai.Client(api_key=key)

    def ask(prompt, question, mode):
        response = client.models.generate_content(model=model_name, contents=prompt,
                                                  config={"temperature": 0})
        return response.text or ""
    return ask


def run(model_name, docs, modes):
    with open(QUESTIONS) as f:
        questions = json.load(f)["questions"]
    if model_name == "mock-oracle":
        model, kind = oracle_model, "mock"
    elif model_name == "mock-flawed":
        model, kind = flawed_model, "mock"
    else:
        model, kind = gemini_model(model_name.split(":", 1)[1]), "real"

    report = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model_name,
              "model_kind": kind, "max_input_chars": MAX_INPUT_CHARS, "documents": {}, "modes": {}}
    scores = {mode: [] for mode in modes}
    extraction = []

    for doc_id in docs:
        try:
            data = load_document(doc_id)
        except Exception as e:  # real papers need network access the first time
            report["documents"][doc_id] = {"skipped": str(e)}
            continue
        result = extract_pdf(data)
        pages, text = result["pages"], "\n".join(result["pages"])
        _, included = build_context(pages, MAX_INPUT_CHARS)
        report["documents"][doc_id] = {
            "pages": len(pages), "chars": len(text), "share_of_input_limit": round(len(text) / MAX_INPUT_CHARS, 3),
            "pages_sent_to_model": included, "extraction_warnings": extraction_warnings(result["report"]),
            "tables_detected": len(result["tables"]),
        }
        for q in (q for q in questions if q["doc"] == doc_id):
            for check in extraction_check(q, pages, result["report"]):
                extraction.append(dict(check, id=q["id"]))
            for mode in modes:
                if mode == "baseline":
                    prompt = baseline_prompt(text[:MAX_INPUT_CHARS], q["question"])
                    scores[mode].append(score_answer(q, model(prompt, q, mode), None, text))
                else:
                    context, _ = build_context(pages, MAX_INPUT_CHARS)
                    prompt = evidence_prompt(context, q["question"])
                    scores[mode].append(score_answer(q, model(prompt, q, mode), pages, text))

    report["extraction"] = {
        "gold_spans": len(extraction),
        "omitted": [e["id"] for e in extraction if not e["extracted"]],
        "omitted_and_unflagged": [e["id"] for e in extraction if not e["flagged_if_missing"]],
    }
    for mode in modes:
        report["modes"][mode] = {"summary": summarize(scores[mode]), "per_question": scores[mode]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mock-oracle", help="mock-oracle, mock-flawed, or gemini:<model>")
    parser.add_argument("--docs", default="all", choices=["all", "synthetic", "real"])
    parser.add_argument("--modes", default="baseline,evidence")
    parser.add_argument("--allow-api-calls", action="store_true", help="required for a real model")
    parser.add_argument("--out", help="write the JSON report here")
    args = parser.parse_args()
    if args.model.startswith("gemini:") and not args.allow_api_calls:
        sys.exit("Real-model runs use your API quota; pass --allow-api-calls to confirm.")

    docs = {"synthetic": list(SYNTHETIC), "real": list(REAL), "all": list(SYNTHETIC) + list(REAL)}[args.docs]
    report = run(args.model, docs, args.modes.split(","))
    output = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
    print(output if not args.out else json.dumps({k: v for k, v in report.items() if k != "modes"} | {
        "summaries": {m: r["summary"] for m, r in report["modes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
