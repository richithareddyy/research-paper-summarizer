"""Evaluation documents.

Synthetic papers are generated from the text below (written for this project, so there are no licensing
limits); their answers are true by construction. Real papers are openly licensed, fetched on demand,
and pinned by SHA-256 so a changed file is noticed rather than silently evaluated.
"""
import hashlib
import os

import pymupdf

from core.net import fetch

# Each page: text, optionally a table (list of rows), optionally "scanned" (rendered as an image with no
# text layer, to model a page that extraction cannot read).
SYNTHETIC = {
    "syn-methods": [
        {"text": "GraphLens: Citation-Aware Retrieval for Scientific Question Answering\n\n"
                 "Abstract. We introduce GraphLens, a retrieval method that uses the citation graph of a paper "
                 "collection to rank evidence passages. GraphLens is evaluated on three question answering "
                 "benchmarks drawn from scientific literature."},
        {"text": "2 Method\n\nThe passage encoder uses 6 transformer layers with a hidden size of 512. "
                 "Citation edges are weighted by the inverse number of references in the citing paper. "
                 "We train with a learning rate of 2e-4 for 40,000 steps on 8 GPUs."},
        {"text": "3 Results\n\nGraphLens reaches 91.2% accuracy on CiteNet. Table 1 lists accuracy on all benchmarks.",
         "table": [["Benchmark", "Accuracy"], ["CiteNet", "91.2"], ["PubBench", "87.5"], ["LegalQA", "79.8"]]},
        {"text": "4 Limitations\n\nWe did not evaluate GraphLens on non-English corpora. The citation graph must be "
                 "available at indexing time, which excludes very recent preprints."},
    ],
    "syn-tables": [
        {"text": "Hyperparameter study for compact summarization models\n\n"
                 "Table 1 reports the training configuration used for every run in this study.",
         "table": [["Setting", "Value"], ["Learning rate", "3e-4"], ["Batch size", "64"], ["Epochs", "20"],
                   ["Warmup steps", "500"]]},
        {"text": "Table 2 reports ROUGE-L for each random seed. The mean ROUGE-L across seeds is 41.7.",
         "table": [["Seed", "ROUGE-L"], ["1", "41.2"], ["2", "42.3"], ["3", "41.6"]]},
    ],
    "syn-scanned": [
        {"text": "A survey of annotated abstract collections\n\n"
                 "This report describes the AbstractBank collection and how it was annotated."},
        {"text": "Collection. The dataset contains 12,480 annotated abstracts drawn from biomedical journals. "
                 "Each abstract was labelled by two annotators.",
         "scanned": True},
        {"text": "Agreement. Inter-annotator agreement measured by Cohen's kappa is 0.81 across all labels."},
    ],
}

REAL = {
    "llama-v1": {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "citation": "Touvron et al., 2023, arXiv:2302.13971v1",
        "url": "https://arxiv.org/pdf/2302.13971v1",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "sha256": "2e663675ae36ad12adb2f5a05281bac2747ecf8d23d92bedd9f937a89fee7136",
    },
}


def _draw_table(page, rows, top):
    left, height = 72, 22
    width = (page.rect.width - 2 * left) / len(rows[0])
    for r in range(len(rows) + 1):
        page.draw_line((left, top + r * height), (left + width * len(rows[0]), top + r * height))
    for c in range(len(rows[0]) + 1):
        page.draw_line((left + c * width, top), (left + c * width, top + len(rows) * height))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((left + c * width + 5, top + r * height + 15), cell, fontsize=10)


def build_synthetic_pdf(doc_id):
    doc = pymupdf.open()
    for spec in SYNTHETIC[doc_id]:
        page = doc.new_page()
        body = pymupdf.Rect(72, 72, page.rect.width - 72, 360)
        if spec.get("scanned"):
            scratch = pymupdf.open()
            scratch_page = scratch.new_page()
            scratch_page.insert_textbox(body, spec["text"], fontsize=11)
            page.insert_image(page.rect, pixmap=scratch_page.get_pixmap(dpi=110))
        else:
            page.insert_textbox(body, spec["text"], fontsize=11)
        if spec.get("table"):
            _draw_table(page, spec["table"], top=380)
    return doc.tobytes()


def load_document(doc_id, cache_dir=".eval-cache"):
    """PDF bytes for a corpus document. Real papers need network access the first time."""
    if doc_id in SYNTHETIC:
        return build_synthetic_pdf(doc_id)
    spec = REAL[doc_id]
    path = os.path.join(cache_dir, f"{doc_id}.pdf")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
    else:
        data = fetch(spec["url"], max_bytes=50 * 1024 * 1024, timeout=120,
                     headers={"User-Agent": "ResearchPaperSummarizer-eval/1.0"}).content
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"{doc_id}: downloaded file does not match the pinned SHA-256 ({digest})")
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data
