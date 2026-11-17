# Research Paper Summarizer

You’re reading a paper and want a quick overview before working through the details, or you have a question about a section you just read. This Streamlit app takes a PDF, pasted text, or DOI/arXiv link and lets you generate summaries, ask follow-up questions, and export notes. The paper’s extracted text, figures, and tables stay available while you read the model’s answers.

**Python · Streamlit · Gemini · PyMuPDF**

**App:** https://researchpapersummerizer-3bmsmrmm7glsxfeavslfb6.streamlit.app/

PyMuPDF extracts the paper’s contents, and Gemini generates summaries, answers, and study aids. This is a reading aid: generated explanations still need checking against the paper.

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

The implementation lives in [`app.py`](app.py). It sends extracted text directly to Gemini, up to a configured character limit; it does not search a separate document index. Multi-paper comparisons use summaries of each paper. Results stay in the current session unless you export them.

## Limitations and checks

Scanned PDFs need text recognition, which this app does not include. DOI lookup may return only an abstract when full text is unavailable. Complex page layouts can confuse extraction. Paper text is sent to the configured model service, whose API quotas apply.

Chat answers quote the passages they rely on, and each quote is checked against the extracted text. Pages the app couldn't read (for example scanned ones) are reported rather than silently skipped. `python -m pytest` runs the deterministic test suite; `eval/` contains a small labelled corpus and harness, but no model-quality evaluation has been run yet. See the guide for what was measured.

[Configuration, workflow, and technical notes](GUIDE.md)
