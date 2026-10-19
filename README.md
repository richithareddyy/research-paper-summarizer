# Research Paper Summarizer

Read a paper, ask questions about it, and export your notes from one Streamlit app. Inputs include PDFs, pasted text, and DOI/arXiv links.

**Python · Streamlit · Gemini · PyMuPDF**

PyMuPDF extracts the paper’s text, embedded figures, and tables. Gemini handles summaries, paper chat, and study aids. The interface keeps those results together so you can return to the source while reading an answer.

## Run locally

Requires Python 3.10+ and a Gemini API key for model features.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Set `GOOGLE_API_KEY` in `.env`. On Windows, activate with `.venv\Scripts\activate`. Open http://localhost:8501 and choose **Try a sample paper** to start without uploading a file. On Streamlit Community Cloud, configure the key in app secrets.

## What’s included

- Five summary styles, follow-up chat, and methodology analysis.
- Figures, tables, references, and TF-IDF keywords.
- Glossaries, flashcards, citations, and multi-paper comparisons.
- Markdown, PDF, and JSON exports.

The implementation lives in [`app.py`](app.py). Paper text is sent directly to Gemini with a character cap; there is no retrieval index. Comparisons use intermediate summaries, and results stay in Streamlit session state unless exported.

## Limitations and checks

There is no OCR for scanned PDFs. DOI lookup may return only an abstract when open full text is unavailable. Extraction depends on layout, and generated answers and citations need checking against the paper. API quotas apply; paper text is shared with the configured model service.

No automated test suite or accuracy evaluation is committed. Check extraction, a question with a known answer, sample-paper comparison, and each export format when making changes.

[Configuration, workflow, and technical notes](GUIDE.md)
