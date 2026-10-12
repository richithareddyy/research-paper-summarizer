import csv
import hashlib
import html
import io
import itertools
import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import quote

import markdown
import pandas as pd
import plotly.express as px
import pymupdf
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

load_dotenv()

APP_TITLE = "Advanced Research Paper Summarizer"
# Mirrors the palette in .streamlit/config.toml for the custom CSS and charts
COLORS = {
    "ink": "#23272F",
    "ink_hover": "#3A404B",
    "accent": "#9A3F1C",
    "text": "#1D2127",
    "text_muted": "#646A73",
    "rule": "#E4E0D8",
    "tag": "#EEEBE4",
    "highlight": "#FCE9A6",
    "grid": "#EFECE6",
    "data": "#56606E",  # chart bars stay neutral so the accent color remains rare
}

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
# Used for quick background calls, e.g. reading a paper's title and authors
FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "gemini-flash-lite-latest")
# Tried in order when the selected model is out of quota or overloaded
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
# Hidden from the model picker: previews often have no free quota, the rest aren't text models
SERVER_ERRORS = (500, 502, 503, 504)
HIDDEN_MODEL_TAGS = ("preview", "exp", "tts", "image", "live", "audio", "embedding", "robotics", "computer-use")
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "400000"))
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "50"))
REQUEST_TIMEOUT = 30
HISTORY_LIMIT = 10
MIN_TEXT_CHARS = 200
MIN_FIGURE_SIZE = 80  # px; smaller images are usually logos or icons
MAX_FIGURES = 60

SUMMARY_TYPES = {
    "comprehensive": "Comprehensive",
    "executive": "Executive",
    "technical": "Technical",
    "critique": "Critique",
    "eli5": "Explain Like I'm 5",
}

ANALYSIS_TYPES = {
    "methodology": "Methodology",
    "literature": "Literature Context",
    "future_research": "Future Research",
    "practical_applications": "Practical Applications",
}

CITATION_STYLES = ["APA", "MLA", "BibTeX"]
NAME_PARTICLES = {"van", "von", "de", "der", "den", "da", "di", "del", "la", "le", "du", "dos", "das", "bin", "al"}

CHAT_HISTORY_TURNS = 6  # earlier exchanges sent with each question for context
CHAT_STARTERS = [
    "What problem does this paper solve?",
    "What are the main results?",
    "What are the limitations?",
]

INPUT_METHODS = {"pdf": "PDF", "text": "Text", "lookup": "DOI / arXiv", "compare": "Compare"}

# Optional sections, toggled in the sidebar. The first four match ANALYSIS_TYPES.
SECTIONS = {
    "methodology": "Methodology",
    "literature": "Literature context",
    "future_research": "Future research",
    "practical_applications": "Practical applications",
    "keywords": "Key terms",
    "figures": "Figures & tables",
    "citations": "Citations by year",
}

SAMPLE_PAPER = {"arxiv_id": "1706.03762", "label": "Attention Is All You Need (2017)"}
SAMPLE_COMPARISON = {"arxiv_ids": ["1706.03762", "1810.04805"], "label": "Attention Is All You Need vs. BERT"}

COMPARISON_FOCUS = {
    "Full Comparison": "",
    "Methodology Comparison": "Concentrate on how the research designs, data, and methods differ and which is more rigorous.",
    "Results Comparison": "Concentrate on the findings: where results agree, where they conflict, and how strong the evidence is.",
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="auto"  # open on desktop, collapsed on phones
)

SESSION_DEFAULTS = {
    "history": [],
    "extracted_text": "",
    "processed_papers": [],
    "current_summary": "",
    "summary_info": "",
    "paper_metadata": {},
    "figures": [],
    "tables": [],
    "references": [],
    "paper_id": None,
    "paper_source": "",
    "analyses": {},
    "follow_up_questions": "",
    "comparison": "",
    "chat_messages": [],
    "sample_comparison": [],
    "study_aids": None,
    "last_upload_id": None,
    "paper_cache": {},
    "input_method": "pdf",
}
for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default.copy() if isinstance(default, (list, dict)) else default

# Styling the theme in .streamlit/config.toml can't express
st.markdown(f"""
<style>
    :root {{
        --rp-ink: {COLORS["ink"]};
        --rp-ink-hover: {COLORS["ink_hover"]};
        --rp-accent: {COLORS["accent"]};
        --rp-text: {COLORS["text"]};
        --rp-muted: {COLORS["text_muted"]};
        --rp-rule: {COLORS["rule"]};
        --rp-tag: {COLORS["tag"]};
        --rp-highlight: {COLORS["highlight"]};
        --rp-serif: "Source Serif 4", Georgia, "Times New Roman", serif;
        --rp-ease: 120ms ease;
    }}

    .block-container {{ padding-top: 1.75rem; padding-bottom: 2rem; max-width: 1320px; }}
    ::selection {{ background: var(--rp-highlight); color: var(--rp-text); }}

    /* Masthead */
    .masthead {{ display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
                 border-bottom: 1px solid var(--rp-text); padding-bottom: 0.6rem; margin-bottom: 1.5rem; }}
    .masthead-title {{ font-family: var(--rp-serif); font-size: clamp(1.5rem, 2.2vw, 1.9rem); font-weight: 600;
                       line-height: 1.2; color: var(--rp-text); margin: 0; padding: 0; }}
    .masthead-subtitle {{ color: var(--rp-muted); font-size: 0.92rem; }}

    /* Numbered section headings */
    .section-heading {{ display: flex; align-items: baseline; gap: 0.6rem;
                        border-bottom: 1px solid var(--rp-rule); padding-bottom: 0.4rem; margin: 0 0 0.9rem; }}
    .section-number {{ font-family: var(--rp-serif); color: var(--rp-accent); font-size: 1rem; font-weight: 600; }}
    .section-title {{ font-family: var(--rp-serif); font-size: 1.3rem; font-weight: 600; color: var(--rp-text); }}

    /* Loaded paper */
    .paper-head {{ margin: 0 0 0.75rem; }}
    .paper-title {{ font-family: var(--rp-serif); font-size: 1.2rem; font-weight: 600; line-height: 1.35;
                    color: var(--rp-text); margin-bottom: 0.2rem; }}
    .paper-byline {{ color: var(--rp-text); font-size: 0.92rem; }}
    .paper-meta {{ color: var(--rp-muted); font-size: 0.85rem; margin-top: 0.15rem; }}
    .paper-meta a {{ color: var(--rp-accent); text-decoration: none; }}
    .paper-meta a:hover {{ text-decoration: underline; }}

    /* Reading width for generated text */
    [class*="st-key-reading"] {{ max-width: 46rem; }}
    [class*="st-key-reading"] p, [class*="st-key-reading"] li {{ line-height: 1.65; }}

    .margin-note {{ border-left: 2px solid var(--rp-accent); padding: 0.1rem 0 0.1rem 0.8rem;
                    color: var(--rp-muted); margin: 0.25rem 0 1rem; font-size: 0.95rem; }}
    .margin-note strong {{ color: var(--rp-text); font-weight: 600; display: block; margin-bottom: 0.1rem; }}
    .key-terms {{ color: var(--rp-muted); font-size: 0.92rem; margin: 0.25rem 0 1rem; }}
    .key-terms span {{ color: var(--rp-text); }}

    /* Sidebar */
    .side-label {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; font-weight: 600;
                   color: var(--rp-muted); margin: 0.75rem 0 0.35rem; }}
    [data-testid="stSidebar"] [class*="st-key-history-"] button {{ justify-content: flex-start; text-align: left;
        padding: 0.2rem 0; min-height: 0; width: 100%; color: var(--rp-text); font-weight: 400; }}
    [data-testid="stSidebar"] [class*="st-key-history-"] button:hover {{ color: var(--rp-accent); }}
    [data-testid="stSidebar"] [class*="st-key-history-"] button:disabled {{ color: var(--rp-accent); opacity: 1; font-weight: 500; }}
    [data-testid="stSidebar"] [class*="st-key-history-"] p {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

    /* Buttons */
    [data-testid^="stBaseButton"] {{ font-weight: 500;
        transition: background-color var(--rp-ease), border-color var(--rp-ease), color var(--rp-ease); }}
    [data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {{
        background-color: var(--rp-ink-hover); border-color: var(--rp-ink-hover); }}
    [data-testid="stBaseButton-secondary"]:hover, [data-testid="stBaseButton-secondaryFormSubmit"]:hover {{
        border-color: var(--rp-ink); color: var(--rp-ink); }}
    [data-testid="stBaseButton-tertiary"]:hover {{ color: var(--rp-accent); }}
    button:disabled {{ cursor: not-allowed; }}
    /* Streamlit's "Press Enter to submit form" hint overlaps long DOIs; the caption says it instead */
    [data-testid="InputInstructions"] {{ display: none; }}
    [class*="st-key-link-"] button {{ color: var(--rp-accent); padding-left: 0; }}
    [class*="st-key-link-"] button p {{ white-space: normal; text-align: left; }}
    [class*="st-key-link-"] button:hover {{ text-decoration: underline; }}

    [data-testid="stFileUploaderDropzone"] {{ padding-top: 1.5rem; padding-bottom: 1.5rem; }}

    [data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] [data-tag] {{
        background-color: var(--rp-tag) !important; color: var(--rp-text) !important;
        border-radius: 3px !important; font-weight: 500; }}
    [data-baseweb="tag"] svg, [data-testid="stMultiSelectTagsContainer"] [data-tag] svg {{
        fill: var(--rp-muted) !important; color: var(--rp-muted) !important; }}

    /* Let segmented controls wrap in narrow columns instead of clipping */
    [data-testid="stButtonGroup"] > div {{ flex-wrap: wrap; }}

    [data-baseweb="tab"] {{ font-weight: 500; transition: color var(--rp-ease); }}
    [data-baseweb="tab"]:hover {{ color: var(--rp-ink); }}
    [data-baseweb="tab-list"] {{ overflow-x: auto; scrollbar-width: thin; }}

    button:focus-visible, a:focus-visible, summary:focus-visible, [role="tab"]:focus-visible,
    input:focus-visible, textarea:focus-visible, [data-tag]:focus-visible {{
        outline: 2px solid var(--rp-accent) !important; outline-offset: 2px; box-shadow: none !important;
    }}

    .app-footer {{ color: var(--rp-muted); font-size: 0.8rem; margin-top: 2rem;
                   padding-top: 0.75rem; border-top: 1px solid var(--rp-rule); }}

    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{ transition: none !important; }}
    }}
    @media (max-width: 640px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 3.5rem; }}
    }}
</style>
""", unsafe_allow_html=True)


def section_heading(number, title):
    st.markdown(
        f'<div class="section-heading" role="heading" aria-level="2">'
        f'<span class="section-number">{number}</span><span class="section-title">{html.escape(title)}</span></div>',
        unsafe_allow_html=True
    )


def margin_note(title, body):
    st.markdown(
        f'<div class="margin-note"><strong>{html.escape(title)}</strong>{html.escape(body)}</div>',
        unsafe_allow_html=True
    )


def render_paper_head():
    metadata = st.session_state.paper_metadata
    title = metadata.get("title") or st.session_state.paper_source or "Untitled paper"
    authors = metadata.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    byline = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    details = [str(v) for v in (metadata.get("publication_year"), metadata.get("journal_or_conference")) if v]
    if metadata.get("doi"):
        doi = html.escape(metadata["doi"])
        details.append(f'<a href="https://doi.org/{doi}" target="_blank">doi:{doi}</a>')
    counts = f"{len(st.session_state.extracted_text.split()):,} words"
    for label, items in (("figures", st.session_state.figures), ("tables", st.session_state.tables),
                         ("references", st.session_state.references)):
        if items:
            counts += f" · {len(items)} {label}"
    st.markdown(
        f'<div class="paper-head"><div class="paper-title">{html.escape(title)}</div>'
        + (f'<div class="paper-byline">{html.escape(byline)}</div>' if byline else "")
        + f'<div class="paper-meta">{" · ".join(details + [counts])}</div></div>',
        unsafe_allow_html=True
    )


def side_label(text):
    st.sidebar.markdown(f'<div class="side-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        font={"family": "Inter, sans-serif", "color": COLORS["text"]},
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


EMPTY_RESPONSE = ("The model returned an empty response. It may have been blocked by safety filters; "
                  "try another summary type or model.")


class GeminiError(Exception):
    """Raised when a Gemini request cannot be completed."""


class PaperLookupError(Exception):
    """Raised when a DOI / arXiv lookup fails."""


def extract_text_from_pdf(pdf_bytes):
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except pymupdf.FileDataError as e:
        raise ValueError(f"The file could not be read as a PDF ({e}).") from e

    with doc:
        if doc.needs_pass:
            raise ValueError("This PDF is password-protected. Please upload an unlocked copy.")

        metadata = {}
        pdf_meta = doc.metadata or {}
        if pdf_meta.get("title", "").strip():
            metadata["title"] = pdf_meta["title"].strip()
        if pdf_meta.get("author", "").strip():
            metadata["authors"] = [a.strip() for a in re.split(r";|\band\b", pdf_meta["author"]) if a.strip()]

        text = "\n".join(page.get_text() for page in doc)

        figures = []
        seen_xrefs = set()
        for page_num, page in enumerate(doc):
            for image in page.get_images(full=True):
                xref = image[0]
                if xref in seen_xrefs or len(figures) >= MAX_FIGURES:
                    continue
                seen_xrefs.add(xref)
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.width < MIN_FIGURE_SIZE or pix.height < MIN_FIGURE_SIZE:
                        continue
                    if pix.n - pix.alpha >= 4:  # CMYK and similar -> RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    figures.append({
                        "page": page_num + 1,
                        "data": pix.tobytes("png"),
                        "width": pix.width,
                        "height": pix.height
                    })
                except Exception:
                    continue  # skip images in formats PyMuPDF can't convert

        tables = []
        for page_num, page in enumerate(doc):
            try:
                found = page.find_tables()
            except Exception:  # table detection is best-effort
                continue
            for table in found.tables:
                rows = [["" if cell is None else str(cell) for cell in row] for row in table.extract()]
                if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    tables.append({"page": page_num + 1, "rows": rows})

    return text, metadata, figures, tables


def extract_references(text):
    heading = re.compile(
        r'^\s*(?:\d+\.?\s*)?(references|bibliography|works cited|literature cited)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    matches = list(heading.finditer(text))
    if not matches:
        matches = list(re.finditer(r'references|bibliography|works cited', text, re.IGNORECASE))
    if not matches:
        return []

    # The last match is most likely the actual references section
    ref_text = text[matches[-1].end():]

    if re.search(r'^\s*\[\d+\]', ref_text, re.MULTILINE):
        entries = re.split(r'^\s*\[\d+\]\s*', ref_text, flags=re.MULTILINE)
    elif re.search(r'^\s*\d+\.\s+\S', ref_text, re.MULTILINE):
        entries = re.split(r'^\s*\d+\.\s+', ref_text, flags=re.MULTILINE)
    else:
        # Author-year style: a new entry starts with "Surname, X"
        entries = re.split(r"\n(?=[A-Z][A-Za-z'\-]+,\s+[A-Z])", ref_text)

    references = [re.sub(r'\s+', ' ', entry).strip() for entry in entries]
    return [ref for ref in references if len(ref) > 20][:500]


def prepare_text_for_model(text):
    """Normalize whitespace and cap the length. Returns (text, truncated)."""
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # re-join words hyphenated across lines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    if len(text) > MAX_INPUT_CHARS:
        return text[:MAX_INPUT_CHARS], True
    return text, False


def paper_id_for(content):
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def safe_filename(title, default="research_paper"):
    safe = re.sub(r'[^\w\s-]', '', title or "").strip()
    safe = re.sub(r'\s+', '_', safe)[:80]
    return safe or default


def format_authors(metadata):
    authors = metadata.get("authors") or []
    if isinstance(authors, str):
        return authors
    return ", ".join(authors) if authors else "Unknown"


@st.cache_data(show_spinner=False)
def extract_keywords(text, top_n=10):
    # TF-IDF over ~80-word passages, so terms concentrated in a few passages
    # outrank words that appear everywhere in the paper
    if not text or len(text) < 100:
        return []

    words = re.sub(r'[^A-Za-z\s-]', ' ', text).split()
    passages = [" ".join(words[i:i + 80]) for i in range(0, len(words), 80)]

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2 if len(passages) >= 4 else 1,
        max_df=0.9 if len(passages) >= 4 else 1.0,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b"
    )

    try:
        matrix = vectorizer.fit_transform(passages)
    except ValueError:  # no usable terms (e.g. no passages, or everything was a stop word)
        return []

    scores = zip(vectorizer.get_feature_names_out(), matrix.sum(axis=0).A1)
    ranked = sorted(scores, key=lambda pair: pair[1], reverse=True)
    return [(word, float(score)) for word, score in ranked[:top_n]]


def citation_years(references):
    """Counts of cited works per publication year, or None if there's too little to plot."""
    if len(references) < 3:
        return None

    current_year = datetime.now().year
    year_counts = {}
    for ref in references:
        for match in re.finditer(r'\b(19[5-9]\d|20\d{2})\b', ref):
            year = int(match.group(1))
            if year <= current_year:
                year_counts[year] = year_counts.get(year, 0) + 1
                break

    if not year_counts:
        return None
    return sorted(year_counts.items())


def get_api_key():
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip()
    try:
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:  # no secrets.toml; the exception type differs across Streamlit versions
        return None


@st.cache_resource(show_spinner=False)
def get_client(api_key):
    # Retry brief server hiccups once. Quota errors (429) aren't retried here because
    # _with_fallback moves on to a backup model instead.
    retry = types.HttpRetryOptions(attempts=2, initial_delay=2, max_delay=5, http_status_codes=list(SERVER_ERRORS))
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=retry))


@st.cache_data(ttl=3600, show_spinner=False)
def list_available_models(api_key):
    names = set()
    for model in get_client(api_key).models.list():
        name = (model.name or "").removeprefix("models/")
        actions = model.supported_actions or []
        if (name.startswith("gemini") and "generateContent" in actions
                and not any(tag in name for tag in HIDDEN_MODEL_TAGS)):
            names.add(name)
    # Newest names first, with the "-latest" aliases (which always point at a current model) on top
    return sorted(sorted(names, reverse=True), key=lambda n: not n.endswith("-latest"))


def _describe_api_error(error, model_name):
    code = getattr(error, "code", None)
    message = getattr(error, "message", None) or str(error)
    if code in (401, 403) or "API key not valid" in message:
        return "The Google API key was rejected. Check GOOGLE_API_KEY in your .env file (or the app's Secrets when deployed)."
    if code == 404:
        return f"The model {model_name} is not available: {message}. Choose a different model in the sidebar."
    if code == 429:
        if re.search(r"limit:\s*0\b", message):
            return (f"Your API key has no free quota for {model_name}. Pick another model in the sidebar, "
                    "or enable billing in Google AI Studio.")
        if re.search(r"per ?day", message, re.IGNORECASE):
            return (f"Your daily free quota for {model_name} is used up. It resets at midnight Pacific time. "
                    "Each model has its own quota, so another model in the sidebar may still work.")
        wait = re.search(r"retry in ([\d.]+)s", message, re.IGNORECASE)
        wait_text = f"about {round(float(wait.group(1)))} seconds" if wait else "a minute"
        return (f"Too many requests to {model_name} in a short time (free-tier rate limit). "
                f"Wait {wait_text} and try again.")
    if code in SERVER_ERRORS:
        return ("The selected Gemini model is busy right now (a temporary problem on Google's side). "
                "Try again in a minute, or pick a different model in the sidebar.")
    return f"Gemini API error ({code}): {message}"


def _with_fallback(model_name, send):
    """Call send(client, model), moving on to FALLBACK_MODELS on quota or availability errors."""
    api_key = get_api_key()
    if not api_key:
        raise GeminiError("No Google API key configured. Add GOOGLE_API_KEY to your .env file (or the app's Secrets when deployed).")

    model_name = model_name.removeprefix("models/")
    candidates = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    first_error = None
    for candidate in candidates:
        try:
            result = send(get_client(api_key), candidate)
        except genai_errors.APIError as e:
            # Out of quota, overloaded, or unavailable: try the next model
            if getattr(e, "code", None) in (404, 429, *SERVER_ERRORS):
                first_error = first_error or e
                continue
            raise GeminiError(_describe_api_error(e, candidate)) from e
        except Exception as e:  # network failures surface as httpx errors from inside the SDK
            raise GeminiError(f"The Gemini request failed: {e}") from e
        if candidate != model_name:
            st.toast(f"{model_name} was unavailable, so {candidate} was used instead.")
        return result

    message = _describe_api_error(first_error, model_name)
    if len(candidates) > 1:
        message += (f" The backup models ({', '.join(candidates[1:])}) were unavailable too, so this is "
                    "most likely a limit on your API key rather than on one model.")
    raise GeminiError(message) from first_error


def call_gemini(prompt, model_name, json_output=False):
    config = types.GenerateContentConfig(response_mime_type="application/json") if json_output else None
    response = _with_fallback(
        model_name, lambda client, model: client.models.generate_content(model=model, contents=prompt, config=config)
    )
    text = response.text
    if not text or not text.strip():
        raise GeminiError(EMPTY_RESPONSE)
    return text.strip()


def stream_gemini(prompt, model_name):
    def start(client, model):
        chunks = iter(client.models.generate_content_stream(model=model, contents=prompt))
        return chunks, next(chunks, None)  # the request is sent here, so errors can trigger a fallback

    chunks, first = _with_fallback(model_name, start)

    def text_chunks():
        try:
            for chunk in itertools.chain([first] if first else [], chunks):
                if chunk.text:
                    yield chunk.text
        except genai_errors.APIError as e:
            raise GeminiError(_describe_api_error(e, model_name)) from e
        except Exception as e:
            raise GeminiError(f"The response was interrupted: {e}") from e

    return text_chunks()


def _ask(prompt, model_name, stream):
    return stream_gemini(prompt, model_name) if stream else call_gemini(prompt, model_name)


def generate_summary(text, model_name=DEFAULT_MODEL, summary_type="comprehensive", stream=False):
    text, _ = prepare_text_for_model(text)

    prompts = {
        "comprehensive": f"""
        You are a research assistant specialized in summarizing academic papers.
        Provide a comprehensive summary of the following research paper:

        <paper>
        {text}
        </paper>

        Structure your summary as follows:
        # Paper Summary
        ## Title and Authors (if available)
        ## Research Question & Objectives
        ## Methodology
        ## Key Findings
        ## Conclusions & Implications

        Make sure to highlight the most important contributions and innovations.
        Format your response in Markdown.
        """,

        "executive": f"""
        Provide a concise executive summary (250-350 words) of the following research paper,
        focusing on the problem addressed, key findings, and practical implications:

        <paper>
        {text}
        </paper>

        Format your response in Markdown.
        """,

        "technical": f"""
        Provide a technical summary of the following research paper, focusing on the methodology,
        technical innovations, algorithms, and experimental results:

        <paper>
        {text}
        </paper>

        Structure your summary as follows:
        # Technical Summary
        ## Problem Statement
        ## Technical Approach
        ## Algorithms & Models
        ## Implementation Details
        ## Evaluation Metrics
        ## Results

        Format your response in Markdown.
        """,

        "critique": f"""
        Provide a critical analysis of the following research paper, evaluating its strengths,
        weaknesses, methodological rigor, and contributions to the field:

        <paper>
        {text}
        </paper>

        Structure your critique as follows:
        # Critical Analysis
        ## Overview
        ## Strengths
        ## Limitations & Weaknesses
        ## Methodological Assessment
        ## Significance & Impact
        ## Suggestions for Improvement

        Format your response in Markdown.
        """,

        "eli5": f"""
        Explain the following research paper as if you were explaining it to a 5th grader.
        Use simple language, analogies, and focus on the big picture ideas:

        <paper>
        {text}
        </paper>

        Keep your explanation under 500 words and make it engaging and easy to understand.
        Format your response in Markdown.
        """
    }

    return _ask(prompts[summary_type], model_name, stream)


def generate_detailed_analysis(text, model_name=DEFAULT_MODEL, analysis_type="methodology", stream=False):
    text, _ = prepare_text_for_model(text)

    prompts = {
        "methodology": f"""
        Provide a detailed analysis of the methodology used in this research paper:

        <paper>
        {text}
        </paper>

        Focus on:
        1. Research design and approach
        2. Data collection methods
        3. Analytical techniques
        4. Validity and reliability considerations
        5. Methodological innovations
        6. Limitations of the methodology

        Format your response in Markdown.
        """,

        "literature": f"""
        Analyze how this paper relates to existing literature in the field:

        <paper>
        {text}
        </paper>

        Focus on:
        1. Key works cited and their importance
        2. How this paper builds on previous research
        3. Gaps in literature this paper addresses
        4. Alternative perspectives not considered
        5. Where this paper fits in the broader research landscape

        Format your response in Markdown.
        """,

        "future_research": f"""
        Based on this research paper, identify promising directions for future research:

        <paper>
        {text}
        </paper>

        Consider:
        1. Unanswered questions raised by this paper
        2. Limitations that could be addressed in future work
        3. Potential extensions of the methodology
        4. New hypotheses suggested by the findings
        5. Interdisciplinary connections that could be explored

        Format your response in Markdown.
        """,

        "practical_applications": f"""
        Identify and elaborate on the practical applications of this research:

        <paper>
        {text}
        </paper>

        Focus on:
        1. Industry applications
        2. Policy implications
        3. Practical tools or frameworks that could be developed
        4. Potential beneficiaries of this research
        5. Steps needed to translate this research into practice

        Format your response in Markdown.
        """
    }

    return _ask(prompts[analysis_type], model_name, stream)


def generate_follow_up_questions(text, model_name=DEFAULT_MODEL, stream=False):
    text, _ = prepare_text_for_model(text)

    prompt = f"""
    Read the following research paper and generate 5 insightful follow-up questions that a researcher
    might ask after reading this paper. These questions should probe deeper into the methodology,
    explore limitations, suggest extensions, or connect to broader research themes.

    <paper>
    {text}
    </paper>

    Format your response as a numbered list in Markdown.
    """

    return _ask(prompt, model_name, stream)


def _parse_json_object(raw, what):
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())  # models sometimes wrap JSON in a code fence
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GeminiError(f"The model did not return valid {what} JSON. Please try again.") from e
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise GeminiError(f"The model did not return a {what} object. Please try again.")
    return data


def extract_paper_details(text, model_name=DEFAULT_MODEL):
    prompt = f"""
    Extract the following metadata from the beginning of this research paper:

    <paper>
    {text[:6000]}
    </paper>

    Return a JSON object with these fields:
    - title (string): The paper's title
    - authors (array of strings): List of authors
    - publication_year (integer or null): Year of publication
    - journal_or_conference (string or null): Publication venue
    - keywords (array of strings): Author-provided keywords
    - doi (string or null): DOI if present

    Use null or an empty array for anything that is not stated in the text.
    """

    data = _parse_json_object(call_gemini(prompt, model_name, json_output=True), "metadata")

    year = data.get("publication_year")
    try:
        year = int(year) if year not in (None, "") else None
    except (TypeError, ValueError):
        year = None

    def as_list(value):
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(v) for v in value or [] if str(v).strip()]

    return {
        "title": (data.get("title") or "").strip() or None,
        "authors": as_list(data.get("authors")),
        "publication_year": year,
        "journal_or_conference": data.get("journal_or_conference") or None,
        "keywords": as_list(data.get("keywords")),
        "doi": data.get("doi") or None,
    }


def compare_papers(papers_text, model_name=DEFAULT_MODEL, comparison_type="Full Comparison"):
    if len(papers_text) < 2:
        raise ValueError("Need at least two papers to compare")

    # Summarize each paper first (in parallel) to keep the comparison prompt small
    ctx = get_script_run_ctx()

    def summarize(text):
        add_script_run_ctx(threading.current_thread(), ctx)  # so st.toast works from worker threads
        return generate_summary(text, model_name, "executive")

    with ThreadPoolExecutor(max_workers=min(4, len(papers_text))) as pool:
        briefs = list(pool.map(summarize, papers_text))
    combined_summaries = "\n\n".join(f"Paper {i}:\n{brief}" for i, brief in enumerate(briefs, 1))
    focus = COMPARISON_FOCUS.get(comparison_type, "")

    prompt = f"""
    Compare and contrast the following research papers based on these summaries:

    {combined_summaries}

    {focus}

    Structure your comparison as follows:
    # Comparative Analysis
    ## Research Focus & Objectives
    ## Methodological Approaches
    ## Key Findings
    ## Strengths & Weaknesses
    ## Complementary Insights
    ## Contradictory Claims (if any)
    ## Integration Possibilities

    Format your response in Markdown.
    """

    return call_gemini(prompt, model_name)


def answer_question(text, question, chat_history, model_name=DEFAULT_MODEL, stream=False):
    text, _ = prepare_text_for_model(text)
    recent = chat_history[-CHAT_HISTORY_TURNS * 2:]
    transcript = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent
    ) or "(no earlier messages)"

    prompt = f"""
    You are a research assistant answering questions about the research paper below.
    Base your answer on the paper. If the paper does not contain the answer, say so clearly;
    you may then add brief general context, labelled as coming from outside the paper.
    Be concise and use Markdown where it helps.

    <paper>
    {text}
    </paper>

    <conversation>
    {transcript}
    </conversation>

    Question: {question}
    """

    return _ask(prompt, model_name, stream)


def generate_study_aids(text, model_name=DEFAULT_MODEL):
    text, _ = prepare_text_for_model(text)

    prompt = f"""
    Create study aids for a student reading the following research paper:

    <paper>
    {text}
    </paper>

    Return a JSON object with:
    - glossary: 10 to 12 objects with "term" and "definition". Pick the terms a reader must
      understand; define each in one or two plain sentences, as it is used in this paper.
    - flashcards: 8 to 10 objects with "question" and "answer" that test understanding of the
      paper's problem, method, results, and limitations. Keep answers to one to three sentences.
    """

    data = _parse_json_object(call_gemini(prompt, model_name, json_output=True), "study aids")

    def clean_pairs(items, first, second):
        cleaned = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                a, b = str(item.get(first) or "").strip(), str(item.get(second) or "").strip()
                if a and b:
                    cleaned.append({first: a, second: b})
        return cleaned

    aids = {
        "glossary": clean_pairs(data.get("glossary"), "term", "definition"),
        "flashcards": clean_pairs(data.get("flashcards"), "question", "answer"),
    }
    if not aids["glossary"] and not aids["flashcards"]:
        raise GeminiError("The model did not return usable study aids. Please try again.")
    return aids


def flashcards_to_csv(flashcards):
    # No header row: Anki and Quizlet import plain question,answer pairs
    buffer = io.StringIO()
    csv.writer(buffer).writerows([card["question"], card["answer"]] for card in flashcards)
    return buffer.getvalue()


def _split_name(name):
    """(last, first) from "First Last" or "Last, First"."""
    name = re.sub(r'\s+', ' ', name).strip()
    if "," in name:
        last, first = (part.strip() for part in name.split(",", 1))
        return last, first
    parts = name.split(" ")
    if len(parts) == 1:
        return parts[0], ""
    i = len(parts) - 1
    while i > 1 and parts[i - 1].lower() in NAME_PARTICLES:  # keep "van der" with the surname
        i -= 1
    return " ".join(parts[i:]), " ".join(parts[:i])


def _initials(first):
    initials = []
    for part in first.replace(".", ". ").split():
        pieces = [p for p in part.split("-") if p and p[0].isalpha()]
        if pieces:
            initials.append("-".join(f"{p[0].upper()}." for p in pieces))
    return " ".join(initials)


def format_citation(metadata, style):
    authors = metadata.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    names = [_split_name(a) for a in authors if a and a.strip()]
    title = (metadata.get("title") or "Untitled").strip().rstrip(".")
    year = metadata.get("publication_year")
    venue = metadata.get("journal_or_conference")
    doi = metadata.get("doi")
    doi_url = f"https://doi.org/{doi}" if doi else ""

    if style == "APA":
        formatted = [f"{last}, {_initials(first)}" if first else last for last, first in names]
        if len(formatted) > 20:
            author_text = ", ".join(formatted[:19]) + ", ... " + formatted[-1]
        elif len(formatted) > 1:
            author_text = ", ".join(formatted[:-1]) + ", & " + formatted[-1]
        else:
            author_text = formatted[0] if formatted else ""
        year_text = f"({year})" if year else "(n.d.)"
        parts = [f"{author_text} {year_text}. {title}." if author_text else f"{title}. {year_text}."]
        if venue:
            parts.append(f"{venue}.")
        if doi_url:
            parts.append(doi_url)
        return " ".join(parts)

    if style == "MLA":
        if len(names) == 1:
            author_text = f"{names[0][0]}, {names[0][1]}".rstrip(", ")
        elif len(names) == 2:
            author_text = f"{names[0][0]}, {names[0][1]}, and {names[1][1]} {names[1][0]}".replace(" ,", ",").strip()
        elif names:
            author_text = f"{names[0][0]}, {names[0][1]}, et al".rstrip(", ")
        else:
            author_text = ""
        details = ", ".join(str(v) for v in (venue, year, doi_url) if v)
        citation = f'{author_text.rstrip(".")}. "{title}."' if author_text else f'"{title}."'
        return f"{citation} {details}." if details else citation

    # BibTeX
    first_author = re.sub(r'[^A-Za-z]', '', names[0][0]).lower() if names else "paper"
    title_word = next((w.lower() for w in re.findall(r'[A-Za-z]+', title) if len(w) > 3), "")
    key = f"{first_author}{year or ''}{title_word}"
    entry_type = "article" if venue and "preprint" not in venue.lower() else "misc"
    fields = [("title", f"{{{title}}}")]
    if names:
        fields.append(("author", " and ".join(f"{last}, {first}" if first else last for last, first in names)))
    if year:
        fields.append(("year", str(year)))
    if venue:
        fields.append(("journal" if entry_type == "article" else "howpublished", venue))
    if doi:
        fields.append(("doi", doi))
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"


def build_paper(text, metadata, figures=None, tables=None, source=""):
    return {
        "id": paper_id_for(text),
        "text": text,
        "metadata": metadata,
        "figures": figures or [],
        "tables": tables or [],
        "references": extract_references(text),
        "filename": source,
    }


def enrich_metadata(paper):
    # Best effort: the paper still loads if this fails
    if not get_api_key():
        return
    try:
        ai_metadata = extract_paper_details(paper["text"], FAST_MODEL)
    except GeminiError as e:
        st.caption(f"Could not extract paper details automatically: {e}")
        return
    for key, value in ai_metadata.items():
        if value:
            paper["metadata"][key] = value


def process_pdf(pdf_bytes, filename):
    pid = paper_id_for(pdf_bytes)
    cache = st.session_state.paper_cache
    if pid in cache:
        return cache[pid]

    text, metadata, figures, tables = extract_text_from_pdf(pdf_bytes)
    if len(text.strip()) < MIN_TEXT_CHARS:
        raise ValueError("No selectable text was found. The PDF may be a scanned image; OCR is not supported.")

    paper = build_paper(text, metadata, figures, tables, filename)
    enrich_metadata(paper)
    cache[pid] = paper
    return paper


def load_paper(paper):
    st.session_state.paper_id = paper["id"]
    st.session_state.paper_source = paper.get("filename", "")
    st.session_state.extracted_text = paper["text"]
    st.session_state.paper_metadata = dict(paper["metadata"])
    st.session_state.figures = paper["figures"]
    st.session_state.tables = paper["tables"]
    st.session_state.references = paper["references"]
    st.session_state.current_summary = paper.get("summary", "")
    st.session_state.summary_info = paper.get("summary_info", "")
    st.session_state.analyses = dict(paper.get("analyses", {}))
    st.session_state.follow_up_questions = paper.get("follow_up_questions", "")
    st.session_state.chat_messages = list(paper.get("chat_messages", []))
    st.session_state.study_aids = paper.get("study_aids")


def _http_headers(include_contact=True):
    # The optional contact email is only sent to Crossref and arXiv, not to publisher sites
    headers = {"User-Agent": "ResearchPaperSummarizer/1.0"}
    mailto = os.getenv("CROSSREF_MAILTO")
    if mailto and include_contact:
        headers["User-Agent"] += f" (mailto:{mailto})"
    return headers


def _download(url, accept_pdf_only=False, include_contact=False):
    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    with requests.get(url, headers=_http_headers(include_contact), timeout=REQUEST_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > limit:
                raise PaperLookupError(f"The file is larger than {MAX_DOWNLOAD_MB} MB.")
            chunks.append(chunk)
    data = b"".join(chunks)
    if accept_pdf_only and not data.startswith(b"%PDF"):
        raise PaperLookupError("The link did not return a PDF.")
    return data


def fetch_crossref_record(doi):
    try:
        response = requests.get(f"https://api.crossref.org/works/{quote(doi)}",
                                headers=_http_headers(), timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        raise PaperLookupError(f"Could not reach Crossref: {e}") from e
    if response.status_code == 404:
        raise PaperLookupError(f"DOI {doi} was not found in Crossref.")
    if not response.ok:
        raise PaperLookupError(f"Crossref returned an error ({response.status_code}).")

    record = response.json().get("message", {})
    date_parts = (record.get("issued") or {}).get("date-parts") or [[None]]
    metadata = {
        "title": " ".join(record.get("title") or []) or None,
        "authors": [
            " ".join(p for p in (a.get("given"), a.get("family")) if p) or a.get("name", "")
            for a in record.get("author", [])
        ],
        "publication_year": date_parts[0][0] if date_parts and date_parts[0] else None,
        "journal_or_conference": " ".join(record.get("container-title") or []) or None,
        "keywords": record.get("subject") or [],
        "doi": doi,
    }
    return metadata, record


def fetch_semantic_scholar_record(doi):
    headers = _http_headers()
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    try:
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}",
            params={"fields": "title,abstract,authors,year,venue,openAccessPdf"},
            headers=headers, timeout=REQUEST_TIMEOUT
        )
        if not response.ok:  # not found, or the shared rate limit was hit
            return None
        data = response.json()
    except (requests.exceptions.RequestException, ValueError):
        return None
    return {
        "title": data.get("title"),
        "authors": [a.get("name") for a in data.get("authors") or [] if a.get("name")],
        "publication_year": data.get("year"),
        "journal_or_conference": data.get("venue") or None,
        "abstract": (data.get("abstract") or "").strip(),
        "pdf_url": (data.get("openAccessPdf") or {}).get("url") or None,
    }


def fetch_arxiv_metadata(arxiv_id):
    response = requests.get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id},
                            headers=_http_headers(), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    entry = ET.fromstring(response.content).find("atom:entry", ns)
    if entry is None:
        return {}

    def find_text(path):
        node = entry.find(path, ns)
        return re.sub(r'\s+', ' ', node.text).strip() if node is not None and node.text else None

    published = find_text("atom:published")
    return {
        "title": find_text("atom:title"),
        "authors": [re.sub(r'\s+', ' ', n.text).strip() for n in entry.findall("atom:author/atom:name", ns) if n.text],
        "publication_year": int(published[:4]) if published else None,
        "journal_or_conference": find_text("arxiv:journal_ref") or "arXiv preprint",
        "doi": find_text("arxiv:doi") or f"10.48550/arXiv.{re.sub(r'v[0-9]+$', '', arxiv_id)}",
    }


ARXIV_PATTERN = re.compile(
    r'^(?:https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/|arxiv:|10\.48550/arxiv\.)?'
    r'(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)(?:\.pdf)?/?$',
    re.IGNORECASE
)
DOI_PATTERN = re.compile(r'(10\.\d{4,9}/\S+)')


def lookup_paper(identifier):
    identifier = identifier.strip()
    if not identifier:
        raise PaperLookupError("Enter a DOI or arXiv ID.")

    arxiv_match = ARXIV_PATTERN.match(identifier)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        try:
            pdf_bytes = _download(f"https://arxiv.org/pdf/{arxiv_id}", accept_pdf_only=True, include_contact=True)
        except requests.exceptions.RequestException as e:
            raise PaperLookupError(f"Could not download arXiv paper {arxiv_id}: {e}") from e
        paper = process_pdf(pdf_bytes, f"arXiv:{arxiv_id}")
        try:
            metadata = fetch_arxiv_metadata(arxiv_id)
            paper["metadata"].update({k: v for k, v in metadata.items() if v})
        except (requests.exceptions.RequestException, ET.ParseError):
            pass  # the full text is already loaded; metadata is a bonus
        return paper

    doi_match = DOI_PATTERN.search(identifier)
    if not doi_match:
        raise PaperLookupError("That does not look like a DOI (10.xxxx/...) or an arXiv ID (e.g. 1706.03762).")
    doi = doi_match.group(1).rstrip(".,;)")
    try:
        metadata, record = fetch_crossref_record(doi)
        crossref_error = None
    except PaperLookupError as e:
        # Not every DOI is registered with Crossref; Semantic Scholar may still know it
        metadata, record, crossref_error = {"doi": doi, "authors": []}, {}, e

    def open_pdf(url, extra_metadata):
        try:
            paper = process_pdf(_download(url, accept_pdf_only=True), f"doi:{doi}")
        except (requests.exceptions.RequestException, PaperLookupError, ValueError):
            return None
        paper["metadata"].update({k: v for k, v in extra_metadata.items() if v})
        return paper

    for link in record.get("link", []):
        if link.get("content-type") == "application/pdf" and link.get("URL"):
            paper = open_pdf(link["URL"], metadata)
            if paper:
                return paper

    abstract = re.sub(r'<[^>]+>', ' ', record.get("abstract") or "")
    abstract = re.sub(r'\s+', ' ', abstract).strip()

    if not abstract:
        scholar = fetch_semantic_scholar_record(doi)
        if scholar:
            for key in ("title", "authors", "publication_year", "journal_or_conference"):
                if not metadata.get(key):
                    metadata[key] = scholar[key]
            if scholar["pdf_url"]:
                paper = open_pdf(scholar["pdf_url"], metadata)
                if paper:
                    return paper
            abstract = scholar["abstract"]
        elif crossref_error:
            raise crossref_error

    if not abstract:
        raise PaperLookupError(
            "This paper isn't openly available: there's no free PDF or abstract to work from. "
            "Download the PDF from the publisher and add it with the PDF option."
        )
    text = f"{metadata.get('title') or ''}\n\n{', '.join(metadata.get('authors') or [])}\n\nAbstract\n{abstract}"
    paper = build_paper(text, metadata, source=f"doi:{doi}")
    paper["abstract_only"] = True
    return paper


def _demote_headings(text):
    return re.sub(r'^(#{1,5})(\s)', r'#\1\2', text, flags=re.MULTILINE)


def build_report_markdown():
    metadata = st.session_state.paper_metadata
    sections = [
        f"# {metadata.get('title') or 'Research Paper Summary'}",
        f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "## Paper Details",
        f"- **Authors**: {format_authors(metadata)}\n"
        f"- **Year**: {metadata.get('publication_year') or 'Unknown'}\n"
        f"- **Journal/Conference**: {metadata.get('journal_or_conference') or 'Unknown'}\n"
        f"- **DOI**: {metadata.get('doi') or 'Unknown'}",
    ]
    if st.session_state.current_summary:
        sections += ["## Summary", _demote_headings(st.session_state.current_summary)]
    keywords = extract_keywords(st.session_state.extracted_text)
    if keywords:
        sections += ["## Keywords", ", ".join(word for word, _ in keywords)]
    for analysis_type, result in st.session_state.analyses.items():
        sections += [f"## {ANALYSIS_TYPES[analysis_type]} Analysis", _demote_headings(result)]
    if st.session_state.follow_up_questions:
        sections += ["## Follow-up Questions", _demote_headings(st.session_state.follow_up_questions)]
    aids = st.session_state.study_aids
    if aids and aids["glossary"]:
        sections += ["## Glossary", "\n".join(f"- **{g['term']}**: {g['definition']}" for g in aids["glossary"])]
    if aids and aids["flashcards"]:
        sections += ["## Flashcards", "\n\n".join(
            f"**Q{i}. {c['question']}**\n\n{c['answer']}" for i, c in enumerate(aids["flashcards"], 1))]
    if st.session_state.chat_messages:
        sections += ["## Questions & Answers", "\n\n".join(
            f"**Q: {m['content']}**" if m["role"] == "user" else _demote_headings(m["content"])
            for m in st.session_state.chat_messages)]
    sections += ["## Citation", format_citation(metadata, "APA"),
                 "```bibtex\n" + format_citation(metadata, "BibTeX") + "\n```"]
    return "\n\n".join(sections) + "\n"


def has_results():
    return bool(st.session_state.current_summary or st.session_state.analyses
                or st.session_state.follow_up_questions or st.session_state.chat_messages
                or st.session_state.study_aids)


def markdown_to_pdf(markdown_text, title):
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "sane_lists"])
    css = (
        "body {font-family: sans-serif; font-size: 10.5pt; line-height: 1.4;} "
        "h1 {font-size: 18pt;} h2 {font-size: 14pt; margin-top: 12pt;} h3 {font-size: 12pt;} "
        "table {border-collapse: collapse;} td, th {border: 1px solid #999; padding: 3pt;} "
        "code, pre {font-family: monospace; font-size: 9pt;}"
    )
    story = pymupdf.Story(html=body, user_css=css)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    page_rect = pymupdf.paper_rect("letter")
    content_rect = page_rect + (54, 54, -54, -54)
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()

    with pymupdf.open(stream=buffer.getvalue(), filetype="pdf") as doc:
        doc.set_metadata({"title": title, "author": "", "creator": APP_TITLE, "producer": ""})
        return doc.tobytes(garbage=3, deflate=True)


def build_export_json():
    export_data = {
        "metadata": st.session_state.paper_metadata,
        "summary": st.session_state.current_summary,
        "summary_info": st.session_state.summary_info,
        "keywords": [
            {"keyword": word, "score": round(score, 4)}
            for word, score in extract_keywords(st.session_state.extracted_text)
        ],
        "analyses": st.session_state.analyses,
        "follow_up_questions": st.session_state.follow_up_questions,
        "chat": st.session_state.chat_messages,
        "study_aids": st.session_state.study_aids,
        "citations": {style: format_citation(st.session_state.paper_metadata, style) for style in CITATION_STYLES},
        "references": st.session_state.references,
        "tables": st.session_state.tables,
        "export_date": datetime.now().isoformat()
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)


def paper_label(paper):
    return paper["metadata"].get("title") or paper["filename"]


def save_to_history():
    entry = {
        "id": st.session_state.paper_id,
        "text": st.session_state.extracted_text,
        "metadata": dict(st.session_state.paper_metadata),
        "figures": st.session_state.figures,
        "tables": st.session_state.tables,
        "references": st.session_state.references,
        "filename": st.session_state.paper_source,
        "summary": st.session_state.current_summary,
        "summary_info": st.session_state.summary_info,
        "analyses": dict(st.session_state.analyses),
        "follow_up_questions": st.session_state.follow_up_questions,
        "chat_messages": list(st.session_state.chat_messages),
        "study_aids": st.session_state.study_aids,
    }
    history = [h for h in st.session_state.history if h["id"] != entry["id"]]
    st.session_state.history = (history + [entry])[-HISTORY_LIMIT:]


def _clear_history():
    st.session_state.history = []


def render_sidebar():
    api_key = get_api_key()
    with st.sidebar:
        side_label("Model")
        models = []
        if api_key:
            try:
                models = list_available_models(api_key)
            except Exception:  # listing models is optional; the default model still works
                pass
        options = [DEFAULT_MODEL] + [m for m in models if m != DEFAULT_MODEL]
        model = st.selectbox("Gemini model", options, label_visibility="collapsed")
        if not api_key:
            st.warning("No Gemini API key found. Add `GOOGLE_API_KEY` to `.env` "
                       "(or the app's Secrets when deployed) and restart.")
        elif not models:
            st.caption("Model list unavailable; using the default.")
        else:
            st.caption("Busy or over-quota models fall back automatically.")

        with st.expander("Sections"):
            sections = {key for key, label in SECTIONS.items()
                        if st.checkbox(label, value=True, key=f"section_{key}")}

    return model, sections


def render_recent_papers():
    # Drawn after the main content so the list reflects a paper loaded during this run
    with st.sidebar:
        side_label("Recent papers")
        history = st.session_state.history
        if history:
            for entry in reversed(history):
                title = entry["metadata"].get("title") or entry.get("filename") or "Untitled paper"
                st.button(title, key=f"history-{entry['id']}", type="tertiary", help=title,
                          on_click=load_paper, args=(entry,), disabled=entry["id"] == st.session_state.paper_id)
            st.button("Clear history", key="clear-history", type="tertiary", on_click=_clear_history)
        else:
            st.caption("Papers you summarize appear here.")

        st.divider()
        st.caption("Summaries, analyses, and answers come from Google's Gemini. Figures, tables, "
                   "references, and key terms are extracted from the paper itself.")


def load_sample_paper():
    with st.spinner("Downloading the sample paper from arXiv..."):
        try:
            paper = lookup_paper(SAMPLE_PAPER["arxiv_id"])
        except (PaperLookupError, ValueError) as e:
            st.error(f"Could not load the sample paper: {e}")
            return
    load_paper(paper)
    st.rerun()  # redraw the panel without the sample link


def load_sample_papers():
    ids = SAMPLE_COMPARISON["arxiv_ids"]
    samples = []
    for i, arxiv_id in enumerate(ids, 1):
        with st.spinner(f"Downloading sample paper {i} of {len(ids)} from arXiv..."):
            try:
                samples.append(lookup_paper(arxiv_id))
            except (PaperLookupError, ValueError) as e:
                st.error(f"Could not load the sample papers: {e}")
                return
    st.session_state.sample_comparison = samples
    st.rerun()


def _clear_sample_papers():
    st.session_state.sample_comparison = []
    st.session_state.comparison = ""


def _keep_input_method():
    # A segmented control can be deselected; keep the previous choice instead
    if st.session_state.input_method is None:
        st.session_state.input_method = st.session_state.get("last_input_method", "pdf")
    st.session_state.last_input_method = st.session_state.input_method


def render_input_panel():
    section_heading(1, "Paper")
    method = st.segmented_control(
        "Add a paper", list(INPUT_METHODS), format_func=INPUT_METHODS.get,
        key="input_method", on_change=_keep_input_method, label_visibility="collapsed"
    ) or "pdf"

    if method == "pdf":
        upload = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")
        if upload is None:
            st.session_state.last_upload_id = None  # so re-uploading the same file loads it again
        else:
            pdf_bytes = upload.getvalue()
            upload_id = paper_id_for(pdf_bytes)
            # Streamlit reruns the script on every interaction; only load a new upload once
            if upload_id != st.session_state.last_upload_id:
                with st.spinner("Reading the PDF..."):
                    try:
                        paper = process_pdf(pdf_bytes, upload.name)
                    except ValueError as e:
                        st.error(f"Could not read this PDF. {e}")
                        return method
                st.session_state.last_upload_id = upload_id
                load_paper(paper)
                st.toast(f"Loaded {upload.name}")

    elif method == "text":
        with st.form("paste_form", border=False):
            pasted = st.text_area("Paper text", height=260, label_visibility="collapsed",
                                  placeholder="Paste the full text of a paper")
            submitted = st.form_submit_button("Analyze Text", type="primary")
        if submitted:
            if len(pasted.strip()) < MIN_TEXT_CHARS:
                st.warning(f"Paste at least {MIN_TEXT_CHARS} characters.")
            else:
                paper = build_paper(pasted.strip(), {}, source="Pasted text")
                with st.spinner("Reading paper details..."):
                    enrich_metadata(paper)
                load_paper(paper)
                st.toast("Text loaded")

    elif method == "compare":
        uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True,
                                   label_visibility="collapsed")
        papers = []
        for i, upload in enumerate(uploads or [], 1):
            with st.spinner(f"Reading file {i} of {len(uploads)}..."):
                try:
                    papers.append(process_pdf(upload.getvalue(), upload.name))
                except ValueError as e:
                    st.error(f"Could not read {upload.name}. {e}")

        samples = st.session_state.sample_comparison
        if samples:
            st.caption("Sample papers: " + ", ".join(paper_label(p) for p in samples))
            st.button("Remove Sample Papers", type="tertiary", on_click=_clear_sample_papers)
        elif not uploads:
            if st.button(f"Try sample papers: {SAMPLE_COMPARISON['label']}", type="tertiary", key="link-samples",
                         help="Downloads two related papers from arXiv."):
                load_sample_papers()
        papers += samples
        st.session_state.processed_papers = papers

        if papers:
            col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
            chosen = col1.selectbox(f"{len(papers)} papers ready. Analyze one on its own:", range(len(papers)),
                                    format_func=lambda i: f"{i + 1}. {paper_label(papers[i])}")
            if col2.button("Open Paper", width="stretch"):
                load_paper(papers[chosen])
                st.rerun()

    elif method == "lookup":
        with st.form("lookup_form", border=False):
            identifier = st.text_input("DOI or arXiv ID/URL", placeholder="1706.03762 or 10.7717/peerj.4375")
            submitted = st.form_submit_button("Look up", type="primary")
        st.caption("arXiv papers load in full. Other DOIs use an open-access PDF when one exists, "
                   "otherwise the abstract. Press Enter or click Look up.")
        if submitted:
            with st.spinner("Looking up the paper..."):
                try:
                    paper = lookup_paper(identifier)
                except (PaperLookupError, ValueError) as e:
                    st.error(str(e))
                    return method
            load_paper(paper)
            if paper.get("abstract_only"):
                st.warning("Only the abstract was available, so results are limited to it.")
            else:
                st.toast("Paper loaded")

    text = st.session_state.extracted_text
    if not text:
        if method != "compare" and st.button(
                f"Try a sample paper: {SAMPLE_PAPER['label']}", type="tertiary", key="link-sample",
                help="Downloads the paper from arXiv so you can try the app without your own PDF."):
            load_sample_paper()
    else:
        with st.expander("Extracted text"):
            st.text(text[:1500] + ("..." if len(text) > 1500 else ""))
        if prepare_text_for_model(text)[1]:
            st.caption(f"Long paper: only the first {MAX_INPUT_CHARS:,} characters are sent to the model.")

    return method


def run_ai_stream(label, func, *args):
    try:
        with st.spinner(label):
            chunks = func(*args, stream=True)
        text = st.write_stream(chunks)
    except GeminiError as e:
        st.error(str(e))
        return None
    # write_stream returns a list rather than a string when nothing was streamed
    if not isinstance(text, str) or not text.strip():
        st.error(EMPTY_RESPONSE)
        return None
    return text.strip()


def run_ai_task(label, func, *args):
    with st.spinner(label):
        try:
            return func(*args)
        except GeminiError as e:
            st.error(str(e))
            return None


def render_output_panel(model, sections):
    section_heading(2, "Analysis")

    if not st.session_state.extracted_text:
        if st.session_state.processed_papers:
            margin_note("Papers ready", "Open one on its own, or compare them in section 3 below.")
        else:
            margin_note("No paper yet",
                        "Add a paper to begin. Its summary, chat, analysis, figures, and citation appear here.")
        return

    render_paper_head()
    summary_tab, chat_tab, analysis_tab, study_tab, figures_tab, export_tab = st.tabs(
        ["Summary", "Chat", "Analysis", "Study", "Figures & Data", "Export"])
    with summary_tab:
        render_summary(model)
    with chat_tab:
        render_chat(model)
    with analysis_tab:
        render_analysis(model, sections)
    with study_tab:
        render_study_aids(model)
    with figures_tab:
        render_figures(sections)
    with export_tab:
        render_export()


def render_summary(model):
    col1, col2 = st.columns([3, 2], vertical_alignment="bottom")
    summary_type = col1.selectbox("Summary type", list(SUMMARY_TYPES), format_func=SUMMARY_TYPES.get)
    label = "Generate New Summary" if st.session_state.current_summary else "Generate Summary"
    if col2.button(label, type="primary", width="stretch"):
        summary = run_ai_stream("Writing the summary...", generate_summary,
                                st.session_state.extracted_text, model, summary_type)
        if summary:
            st.session_state.current_summary = summary
            st.session_state.summary_info = f"{SUMMARY_TYPES[summary_type]} summary · {model}"
            save_to_history()
            st.rerun()  # refresh the button label and the export tab

    if st.session_state.current_summary:
        with st.container(key="reading-summary"):
            st.caption(st.session_state.summary_info)
            st.markdown(st.session_state.current_summary)


def render_analysis(model, sections):
    st.subheader("Detailed Analysis")
    available = [key for key in ANALYSIS_TYPES if key in sections]
    if available:
        col1, col2 = st.columns([3, 2], vertical_alignment="bottom")
        analysis_type = col1.selectbox("Analysis type", available, format_func=ANALYSIS_TYPES.get)
        if col2.button("Generate Analysis", type="primary", width="stretch"):
            result = run_ai_stream("Analyzing the paper...", generate_detailed_analysis,
                                   st.session_state.extracted_text, model, analysis_type)
            if result:
                st.session_state.analyses[analysis_type] = result
                save_to_history()
                st.rerun()
        for key, result in st.session_state.analyses.items():
            with st.expander(f"{ANALYSIS_TYPES[key]} Analysis", expanded=(key == analysis_type)):
                with st.container(key=f"reading-analysis-{key}"):
                    st.markdown(result)
    else:
        st.caption("Turn on an analysis under Sections in the sidebar.")

    st.subheader("Research Questions")
    if st.button("Generate Follow-up Questions"):
        questions = run_ai_stream("Thinking of questions...", generate_follow_up_questions,
                                  st.session_state.extracted_text, model)
        if questions:
            st.session_state.follow_up_questions = questions
            save_to_history()
            st.rerun()
    if st.session_state.follow_up_questions:
        with st.container(key="reading-questions"):
            st.markdown(st.session_state.follow_up_questions)

    if "keywords" in sections:
        keywords = extract_keywords(st.session_state.extracted_text)
        if keywords:
            st.subheader("Key Terms")
            st.markdown('<div class="key-terms"><span>'
                        + "</span> · <span>".join(html.escape(word) for word, _ in keywords)
                        + "</span></div>", unsafe_allow_html=True)

    st.subheader("Cite This Paper")
    style = st.radio("Citation style", CITATION_STYLES, horizontal=True, label_visibility="collapsed")
    st.code(format_citation(st.session_state.paper_metadata, style), language=None, wrap_lines=True)
    st.caption("Built from the extracted details. Double-check before submitting.")

    if st.session_state.references:
        with st.expander(f"References ({len(st.session_state.references)})"):
            for i, ref in enumerate(st.session_state.references, 1):
                st.markdown(f"{i}. {ref}")


def render_figures(sections):
    views = [label for label, key in (("Top Keywords", "keywords"), ("Citations by Year", "citations"),
                                      ("Figures & Tables", "figures")) if key in sections]
    if not views:
        st.caption("Turn on key terms, citations by year, or figures & tables under Sections in the sidebar.")
        return
    view = st.segmented_control("Show", views, default=views[0], label_visibility="collapsed") or views[0]

    if view == "Top Keywords":
        st.caption("Terms that stand out in this paper, scored with TF-IDF.")
        keywords = extract_keywords(st.session_state.extracted_text, top_n=15)
        if keywords:
            top_score = keywords[0][1]
            fig = px.bar(
                x=[int(score / top_score * 100) for _, score in keywords],
                y=[word for word, _ in keywords],
                orientation='h',
                labels={"x": "Relative Importance", "y": ""},
                color_discrete_sequence=[COLORS["data"]]
            )
            fig.update_layout(height=480, yaxis={"autorange": "reversed"})
            st.plotly_chart(style_chart(fig), width="stretch")
        else:
            st.info("Not enough text to generate keyword visualization")

    elif view == "Citations by Year":
        st.caption("Publication years of the works this paper cites.")
        years = citation_years(st.session_state.references)
        if years:
            fig = px.bar(
                x=[year for year, _ in years],
                y=[count for _, count in years],
                labels={"x": "Year", "y": "Number of Citations"},
                color_discrete_sequence=[COLORS["data"]]
            )
            st.plotly_chart(style_chart(fig), width="stretch")
        else:
            st.info("Not enough references to generate citation graph")

    else:
        st.subheader("Extracted Figures & Tables")
        figures = st.session_state.figures
        if figures:
            st.write(f"Displaying {len(figures)} extracted figures")
            cols = st.columns(2)
            for i, figure in enumerate(figures):
                with cols[i % 2]:
                    st.image(figure["data"], caption=f"Figure from page {figure['page']}", width="stretch")
        else:
            st.info("No figures extracted from this document")

        tables = st.session_state.tables
        if tables:
            st.write(f"Displaying {len(tables)} detected tables")
            for table in tables:
                with st.expander(f"Table from page {table['page']}"):
                    st.dataframe(pd.DataFrame(table["rows"]), hide_index=True)
        else:
            st.info("No tables detected in this document")


def render_export():
    st.subheader("Export Options")
    if not has_results():
        margin_note("Nothing to export yet",
                    "Generate a summary, analysis, chat answer, or study aids first. Everything you create is included.")
        return

    title = st.session_state.paper_metadata.get("title") or "Research Paper Summary"
    base_name = f"{safe_filename(title)}_summary"
    report = build_report_markdown()

    export_format = st.radio("Format", ["Markdown", "PDF", "JSON"], horizontal=True)
    if export_format == "Markdown":
        data, ext, mime = report, "md", "text/markdown"
    elif export_format == "PDF":
        try:
            data, ext, mime = markdown_to_pdf(report, title), "pdf", "application/pdf"
        except (RuntimeError, ValueError) as e:
            st.error(f"Could not create the PDF: {e}")
            data = None
    else:
        data, ext, mime = build_export_json(), "json", "application/json"
    if data is not None:
        st.download_button(f"Download {export_format}", data, file_name=f"{base_name}.{ext}", mime=mime,
                           type="primary", icon=":material/download:", on_click="ignore")

    with st.expander("Preview"):
        st.markdown(report)


def _queue_question(question):
    st.session_state.pending_question = question


def _clear_chat():
    st.session_state.chat_messages = []
    save_to_history()


def render_chat(model):
    messages = st.session_state.chat_messages

    if not messages:
        st.caption("Ask anything about this paper. Answers are based on the paper's text.")
        for question in CHAT_STARTERS:
            st.button(question, on_click=_queue_question, args=(question,), key=f"starter_{question}")

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about this paper") or st.session_state.pop("pending_question", None)
    if question and question.strip():
        question = question.strip()
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer = run_ai_stream("Reading the paper...", answer_question,
                                   st.session_state.extracted_text, question, messages, model)
        if answer:
            st.session_state.chat_messages = messages + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            save_to_history()
            st.rerun()

    if messages:
        st.button("Clear Chat", icon=":material/delete:", type="tertiary", on_click=_clear_chat)


def render_study_aids(model):
    aids = st.session_state.study_aids
    label = "Regenerate Study Aids" if aids else "Generate Study Aids"
    if st.button(label, type="secondary" if aids else "primary"):
        result = run_ai_task("Creating a glossary and flashcards...", generate_study_aids,
                             st.session_state.extracted_text, model)
        if result:
            st.session_state.study_aids = result
            save_to_history()
            st.rerun()

    if not aids:
        st.caption("Creates a glossary of the paper's key terms and flashcards to test your understanding.")
        return

    if aids["glossary"]:
        st.subheader(f"Glossary ({len(aids['glossary'])} terms)")
        for item in aids["glossary"]:
            st.markdown(f"**{item['term']}**: {item['definition']}")

    if aids["flashcards"]:
        st.subheader(f"Flashcards ({len(aids['flashcards'])})")
        st.caption("Click a card to reveal the answer.")
        for i, card in enumerate(aids["flashcards"], 1):
            with st.expander(f"{i}. {card['question']}"):
                st.markdown(card["answer"])
        title = st.session_state.paper_metadata.get("title") or "paper"
        st.download_button("Download Flashcards (CSV)", flashcards_to_csv(aids["flashcards"]),
                           file_name=f"{safe_filename(title)}_flashcards.csv", mime="text/csv",
                           icon=":material/download:", on_click="ignore",
                           help="Question and answer columns; imports into Anki and Quizlet.")


def render_comparison(model):
    papers = st.session_state.processed_papers
    if len(papers) < 2:
        return

    section_heading(3, "Comparison")
    selected = st.multiselect(
        "Papers to compare",
        range(len(papers)),
        default=list(range(len(papers))) if len(papers) <= 3 else None,
        format_func=lambda i: f"{i + 1}. {paper_label(papers[i])}"
    )
    focus = st.radio("Focus", list(COMPARISON_FOCUS), horizontal=True)

    if len(selected) < 2:
        st.caption("Select at least two papers.")
    elif st.button("Compare Selected Papers", type="primary"):
        comparison = run_ai_task(
            f"Comparing {len(selected)} papers (each is summarized first, in parallel)...",
            compare_papers, [papers[i]["text"] for i in selected], model, focus
        )
        if comparison:
            st.session_state.comparison = comparison

    if st.session_state.comparison:
        with st.container(key="reading-comparison"):
            st.markdown(st.session_state.comparison)
        st.download_button("Download Comparison", st.session_state.comparison, file_name="paper_comparison.md",
                           mime="text/markdown", icon=":material/download:", on_click="ignore")


def main():
    st.markdown(
        f'<header class="masthead"><h1 class="masthead-title">{APP_TITLE}</h1>'
        '<span class="masthead-subtitle">Summarize, question, and compare academic papers</span></header>',
        unsafe_allow_html=True
    )

    model, sections = render_sidebar()

    # Results get the wider column; that's where most of the reading happens
    left, right = st.columns([5, 7], gap="large")
    with left:
        method = render_input_panel()
    with right:
        render_output_panel(model, sections)

    if method == "compare":
        render_comparison(model)

    render_recent_papers()

    st.markdown(
        '<footer class="app-footer">Gemini can make mistakes. Check important details against the paper.</footer>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
