# Research Paper Summarizer

You’re reading a paper and want a quick overview before working through the details, or you have a question about a section you just read. This Streamlit app takes a PDF, pasted text, or DOI/arXiv link and lets you generate summaries, ask follow-up questions, and export notes. The paper’s extracted text stays available while you read the model’s answers, along with any figures and tables found in a PDF. The app’s page title is “Advanced Research Paper Summarizer.”

**Python · Streamlit · Gemini · PyMuPDF**

**App:** https://researchpapersummerizer-3bmsmrmm7glsxfeavslfb6.streamlit.app/ (hosted on Streamlit Community Cloud; after a period without visitors it may ask you to wake it)

PyMuPDF extracts the paper’s contents, and Gemini generates summaries, analyses, answers, comparisons, and study aids. Key terms, citation formatting, and charts are computed without the model. This is a reading aid: generated explanations still need checking against the paper.

## Run locally

Requires Python 3.10+. A Gemini API key is needed only for model features; without one you can still load papers and browse their text, figures, tables, references, key terms, citations, and charts.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Set `GOOGLE_API_KEY` in `.env`. On Windows, use `python` instead of `python3` and activate with `.venv\Scripts\activate`. Open http://localhost:8501 and choose **Try a sample paper**, which downloads *Attention Is All You Need* from arXiv, to start without uploading a file. On Streamlit Community Cloud, configure the key in app secrets.

## What’s included

- **Input:** a PDF, pasted text, a DOI or arXiv ID/URL, or several PDFs to compare.
- **Reading:** five summary styles; methodology, literature-context, future-research, and practical-application analyses; follow-up questions; and chat whose quoted evidence is checked against the paper.
- **Extraction:** figures, tables, references, TF-IDF key terms, citations by year, and a viewer for the extracted text (page by page for PDFs).
- **Study and reuse:** a glossary, flashcards (CSV), APA/MLA/BibTeX citations, and multi-paper comparisons (full, methodology, or results).
- **Export and history:** Markdown, PDF, and JSON exports once you have generated something, and the last 10 papers you generated something for in the session.

[`app.py`](app.py) holds the interface, lookups, and most prompts. [`core/`](core) holds PDF extraction, the chat prompt and quote checking, and the code that fetches papers and metadata; [`eval/`](eval) and [`tests/`](tests) hold the evaluation harness and tests. Extracted text goes directly to Gemini, up to a configured character limit (400,000 by default); there is no embedding index or retrieval step. Multi-paper comparisons first summarize each paper and compare those summaries. If the selected Gemini model is unavailable or out of quota, the app tries `gemini-flash-latest` and then `gemini-flash-lite-latest`.

### Where your data goes

- With an API key set, the first 6,000 characters of a PDF or pasted text go to Gemini as soon as it is loaded, to read the title and authors. Text up to the character limit is sent when you use a model feature (summary, analysis, questions, chat, comparison, or study aids).
- DOI and arXiv lookups contact arXiv, Crossref, Semantic Scholar, and any open-access PDF link those services return.
- Papers and results are kept in the Streamlit session and are not written to disk by the app. Extracted key terms are also cached in the server process’s memory, shared by its sessions, until it restarts.

## Limitations and checks

- There is no text recognition (OCR). PDFs that need a password to open, and PDFs or pasted text with fewer than 200 readable characters, are refused.
- Uploads and downloaded PDFs are limited to 50 MB, and 300 pages are read by default.
- DOI lookup may return only an abstract, or nothing when neither an open PDF nor an abstract is available. Complex layouts can confuse extraction.
- For a loaded PDF, the app warns about pages with almost no selectable text, unread pages past the limit, damaged files, images past the figure limit, failed table detection, and text left out of the model’s input. It does not catch everything: a scanned page with a readable header, garbled fractions, undecodable images, and references past 500 are not flagged, and the Compare view does not show these warnings for its PDFs.
- The chat prompt asks the model to quote the passages it relies on. Each quote is matched against the extracted text and labelled as found, found on another page, or not found; an answer without quotes is labelled too. A matched quote shows the text is in the paper, not that the answer is right. Summaries, analyses, comparisons, and study aids are not checked this way.
- Paper and metadata downloads are limited to public `http(s)` addresses, with each redirect rechecked and size and time limits. DNS rebinding is not blocked.
- Gemini API quotas apply.

## Tests and evaluation

```bash
pip install -r requirements-dev.txt
python -m pytest
python -m eval.run --model mock-oracle
```

The tests use a fake Gemini client and make no API calls. `eval/` contains 18 questions over three synthetic papers and one CC BY 4.0 paper (LLaMA, arXiv:2302.13971v1, downloaded on first use). Extraction preserved 12 of the 14 labelled evidence spans; of the two it lost, one was on a scanned page and was flagged, and one was a fraction and was not. The LLaMA labels were drafted with an LLM-based assistant and have not been independently validated. Mock runs check the scoring only; no evaluation has been run with a real model, so there are no accuracy figures.

[Configuration, workflow, and technical notes](GUIDE.md)
