# Research Paper Summarizer

Turn a PDF, pasted text or DOI/arXiv link into structured summaries, methodology insights, a chat you can question, study aids and exportable research notes. Built with **Python · Streamlit · Gemini · PyMuPDF**.

[Get started](#setup) · [Features](#features) · [Configuration](#configuration) · [Limitations](#limitations)

## How the reading workflow is implemented

The goal is to keep reading, asking questions, and taking notes in one place. [`app.py`](app.py) contains the Streamlit interface, lookups, and most prompts; [`core/`](core) contains extraction, the chat prompt and quote checking, and downloads:

- PyMuPDF extracts selectable text page by page, plus embedded images and tables ([`core/pdf.py`](core/pdf.py)). Each extraction produces a report of what could not be read: pages with no text layer, pages beyond `MAX_PAGES`, repaired (damaged) files, images past the figure limit, and failed table detection. For a loaded paper, the app shows these as "Extraction is incomplete" in the Paper panel. Not everything is caught: a scanned page with a readable header, a garbled fraction, an image that cannot be decoded, and references past 500 are not flagged, and the Compare view does not show these warnings for its PDFs.
- DOI lookup tries Crossref and Semantic Scholar; arXiv lookup can fetch a PDF. Paper and metadata requests go through [`core/net.py`](core/net.py) (see *Downloads* below); Gemini calls use Google's SDK.
- Summaries and chat send paper text directly to Gemini. Text is normalized and capped by `MAX_INPUT_CHARS`; there is no embedding index or retrieval layer, and text beyond that cap is omitted (and reported).
- Chat answers are evidence-linked ([`core/evidence.py`](core/evidence.py)): the model sees the paper with `[Page N]` markers and is asked to quote the passages it relies on, or start with "Not stated in the paper." Each quote is then checked against the extracted text and shown as found, found on a different page, or not found, with the surrounding source text when it can be located. An answer without quotes is labelled as such. Pasted text and abstract-only papers have no pages, so quotes are checked against the whole text. The **Source text** viewer shows any processed page as extracted.
- Comparisons first generate an executive summary for each paper, using up to four workers, then compare those summaries. That keeps the comparison prompt smaller but can lose details from the originals.
- Extracted material and generated results stay in Streamlit session state. Extracted key terms are also cached in the server process's memory, shared across sessions, until it restarts. The export functions assemble those results into Markdown, PDF, or JSON.

## At a glance

| Bring a paper | Explore the research | Learn and take it with you |
| --- | --- | --- |
| PDF upload, pasted text, DOI or arXiv lookup, or built-in sample papers | Streaming summaries, chat with the paper, methodology, figures, tables and comparisons | Glossary and flashcards, APA/MLA/BibTeX citations, Markdown, PDF and JSON reports |

## Workflow

1. **Load** a paper, or select **Try a sample paper**.
2. **Choose** a summary type next to **Generate Summary**, and optional sections in the sidebar.
3. **Explore** the summary, chat with the paper, and browse its figures, tables and references.
4. **Study** with the glossary and flashcards, and copy a citation.
5. **Compare or export** findings for further reading. Try **Compare → Try sample papers**
   to compare two papers without your own files.

## Features

- **Input methods**
  - Upload a single PDF
  - Paste text
  - Upload several PDFs and compare them, or load two sample papers to compare
  - Look up a paper by DOI or arXiv ID/URL. arXiv papers are downloaded in full. For other
    DOIs the app uses an open-access PDF from Crossref or Semantic Scholar when one exists, and otherwise the abstract.
- **Summaries**: comprehensive, executive, technical, critique, or "explain like I'm 5", shown as they are written
- **Detailed analysis**: methodology, literature context, future research, and practical applications
- **Follow-up questions**: research questions the paper raises
- **Chat with the paper**: ask questions and get answers based on the paper's text, with starter questions to get going
- **Study aids**: a glossary of key terms and flashcards, downloadable as CSV for Anki or Quizlet
- **Citations**: copy-ready APA, MLA, and BibTeX citations built from the paper's details
- **Metadata**: title, authors, year, venue, and DOI (from the PDF, Gemini, Crossref, Semantic Scholar, or arXiv)
- **Extraction**: references, embedded figures, and tables (via PyMuPDF table detection)
- **Visualizations**: TF-IDF keywords, citations by year, and a figure/table gallery
- **Paper comparison**: full, methodology-focused, or results-focused; papers are summarized in parallel
- **Export**: once you have generated something, a report with the summary, key terms, analyses, questions, chat with its checked quotes, study aids, extraction notes, and citation, as Markdown, PDF, or JSON; JSON also includes references and tables
- **History**: reopen the last 10 papers you generated something for in this session, including their results

Use **Sections** in the sidebar to choose which analyses and visualizations appear.

## Requirements

- Python 3.10 or newer
- For model features, a Google Gemini API key from https://aistudio.google.com/apikey

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: use python, then .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your key in .env
```

## Run

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 and reads your key from `.env` automatically. If no key is
found, the sidebar explains how to add one; you can still load papers and browse their text, figures,
tables, references, key terms, charts, and citations.

To try the app without your own PDF, click **Try a sample paper**. It downloads
*Attention Is All You Need* (Vaswani et al., 2017) from arXiv. Then click **Generate Summary**.

To try a comparison, choose **Compare** and click **Try sample papers**. It downloads
*Attention Is All You Need* and *BERT* (Devlin et al., 2018) from arXiv, ready to compare.

## Configuration

Set these in `.env`. All are optional except `GOOGLE_API_KEY`, which model features need.

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key (`GEMINI_API_KEY` also works) |
| `GEMINI_MODEL` | `gemini-flash-latest` | Default model. The sidebar also lists the Gemini text models your key can use, except preview and experimental ones. |
| `GEMINI_FAST_MODEL` | `gemini-flash-lite-latest` | Faster model for quick background tasks, such as reading a paper's title and authors on upload |
| `MAX_INPUT_CHARS` | `400000` | Maximum characters of paper text in one request to the model |
| `MAX_DOWNLOAD_MB` | `50` | Size limit for PDFs downloaded during DOI/arXiv lookup (the upload limit is set separately) |
| `MAX_PAGES` | `300` | Pages read from a PDF; later pages are reported as not processed |
| `CROSSREF_MAILTO` | — | Optional contact email sent to Crossref, arXiv, and Semantic Scholar with each request |
| `SEMANTIC_SCHOLAR_API_KEY` | — | Optional key for Semantic Scholar, used when Crossref has no abstract or open PDF for a DOI |

For Streamlit Community Cloud, put `GOOGLE_API_KEY` in the app's secrets instead of `.env`.

The app's look (colors, fonts, corner radius, status and chart colors) is set in
`.streamlit/config.toml`. A small CSS block near the top of `app.py` handles what the theme can't,
such as section headings, focus outlines, and the reading width for generated text. Its colors
are in `COLORS`, which repeats the theme's main values and adds a few of its own.

## Limitations

- There is no OCR. PDFs or pasted text with fewer than 200 readable characters are refused, as are PDFs
  that need a password to open.
- Reference parsing, table detection, and citation years use heuristics. Results depend on
  the paper's layout.
- Many publisher DOIs have no open full text. For those, download the PDF and upload it.
- Every AI feature needs a valid Gemini API key and uses your API quota.
- Free-tier keys can have daily and per-minute limits for each model. If the selected model is out of
  quota, overloaded, or unavailable, the app automatically retries with `gemini-flash-latest` and then
  `gemini-flash-lite-latest`. Preview and experimental models are hidden from the model list
  because they often have no free quota.

## Downloads

These rules apply to paper and metadata requests. Fetching papers means requesting URLs that come from third-party metadata (Crossref and Semantic Scholar links), so requests are restricted:

- only `http` and `https`, no embedded credentials;
- the host must resolve only to public addresses (loopback, private, link-local such as cloud metadata at 169.254.169.254, carrier-grade NAT, and multicast are refused);
- redirects are followed manually, up to five, and every hop is checked the same way;
- a byte limit (`MAX_DOWNLOAD_MB` for PDFs, 5 MB for API responses), checked against `Content-Length` and while streaming;
- an overall time budget per request, checked between redirects and as data arrives; a read that stalls can still overrun it by up to its own timeout.

This came out of an audit that found the earlier code followed redirects to any destination; it was a hardening gap, not a demonstrated exploit. One gap remains: the address is checked when the URL is resolved and requests resolves it again when connecting, so a DNS server that changes its answer in between (DNS rebinding) is not blocked. Request headers, including the optional Semantic Scholar key, are also resent on redirects. Uploads are limited to 50 MB in `.streamlit/config.toml`.

## Evaluation

[`eval/`](eval) holds a small labelled corpus and a harness that compares the original full-text chat prompt (*baseline*) with the evidence-linked prompt (*evidence*).

**Corpus** ([`eval/questions.json`](eval/questions.json), 18 questions):

- three synthetic papers generated from text in [`eval/corpus.py`](eval/corpus.py): a methods paper, a table-heavy paper, and one with an image-only (scanned) page;
- *LLaMA: Open and Efficient Foundation Language Models* (arXiv:2302.13971v1, CC BY 4.0), downloaded on first use and pinned by SHA-256; the PDF is not committed.

Questions cover facts with page and span references, table cells, and unanswerable questions. Every label records its provenance. Synthetic labels are true by construction. The LLaMA labels were drafted with an LLM-based assistant and are **not independently validated**; their evidence spans are checked automatically against the extracted text, but the answers and the "unanswerable" judgements have not been reviewed by a person.

**Metrics** ([`eval/metrics.py`](eval/metrics.py)): answer accuracy (pattern match), answer support (correct and backed by at least one verified quote), citation precision (cited quotes found on the cited page; a quote citing no page counts if it is found anywhere), citations on the gold page, abstention recall on unanswerable questions, false abstentions, and extraction omissions (gold spans missing from the extracted text, and whether the extraction report flagged them).

```bash
pip install -r requirements-dev.txt
python -m pytest                                  # no API calls; 63 tests, one skipped until LLaMA is cached
python -m eval.run --model mock-oracle            # harness check with a mock model; downloads LLaMA once
python -m eval.run --model gemini:gemini-flash-latest --allow-api-calls --out results.json
```

**What has actually been measured** (no model involved):

| Document | Pages | Characters | Share of `MAX_INPUT_CHARS` | Gold spans extracted |
|---|---|---|---|---|
| syn-methods / syn-tables | 4 / 2 | 850 / 325 | <1% | all |
| syn-scanned | 3 | 211 | <1% | 1 of 2 missing (scanned page 2), **flagged** |
| LLaMA v1 | 27 | 87,897 | 22% | 6 of 7; the "2/3 4d" fraction is lost (extracted as `2` and `34d`) and **not flagged** |

The audit also found that PyMuPDF's table extraction misplaced decimal points in the LLaMA tables (`67.0%` came out as `670%` and `.` on separate lines). Table cells are now read from each cell's box on the page, which gives the correct values.

**Retrieval was not added.** The longest paper in the corpus uses 22% of the input limit and is sent whole, so there was no truncation for retrieval to fix. Retrieval would be worth measuring for theses or very long reports; the Paper panel and exports already say when pages are left out.

**What has not been measured:** answer quality with a real model. The mock runs only show that the metrics behave as intended. The oracle mock scores 100% on accuracy and abstention and 12 of 14 on support, because the two omitted spans can't be quoted; the flawed mock's planted wrong pages, invented quotes, wrong answers, and missing abstentions are all detected. No Gemini evaluation has been run, so there are no accuracy figures for the baseline or evidence modes. Running one uses your API quota and needs `--allow-api-calls`. With 18 questions, results would be indicative at best.

## Checking a change

`python -m pytest` runs the deterministic suite: extraction edge cases (malformed, truncated, password-protected, owner-only encryption, scanned pages, page limits, table-heavy files), download restrictions, quote checking, evaluation scoring, and end-to-end app checks with a fake Gemini client. The LLaMA span check runs only if the PDF is cached in `.eval-cache/` (`python -m eval.run --docs real` downloads it).

Generated summaries, citations, and study aids still need checking against the paper. A verified quote shows that the quoted text is in the paper, not that the answer's reasoning is correct. Model context is cut at the input limit (at a page boundary for chat about a PDF, otherwise by character) rather than selected for relevance, and multi-paper conclusions are based on intermediate summaries. With an API key set, the first 6,000 characters of a PDF or pasted text are sent to Gemini when it is loaded, to read its title and authors; more is sent when you use a model feature. Prompts that contain paper text mark it as material rather than instructions (the final comparison prompt, built from generated summaries, does not), which reduces but does not eliminate prompt injection from a paper's own text.
