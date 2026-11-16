"""End-to-end checks of the Streamlit app with a fake Gemini client (no network, no API calls)."""
import os
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

from eval.corpus import build_synthetic_pdf

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


class Chunk:
    def __init__(self, text):
        self.text = text


class FakeModels:
    reply = ""

    def list(self):
        return []

    def generate_content(self, model, contents, config=None):
        return Chunk('{"title": "GraphLens", "authors": ["A. Author"]}' if config else FakeModels.reply)

    def generate_content_stream(self, model, contents, config=None):
        FakeModels.last_prompt = contents
        return iter([Chunk(FakeModels.reply)])


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.models = FakeModels()


@pytest.fixture
def app():
    with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
            mock.patch("google.genai.Client", FakeClient):
        at = AppTest.from_file(APP, default_timeout=120).run()
        yield at


def upload(at, doc_id):
    at.file_uploader[0].set_value((f"{doc_id}.pdf", build_synthetic_pdf(doc_id), "application/pdf")).run()


def test_incomplete_extraction_is_shown(app):
    upload(app, "syn-scanned")
    assert not app.exception
    warning = " ".join(w.value for w in app.warning)
    assert "Extraction is incomplete" in warning and "scanned" in warning


def test_complete_extraction_has_no_warning(app):
    upload(app, "syn-methods")
    assert not [w for w in app.warning if "Extraction" in w.value]


def test_chat_answer_shows_checked_evidence(app):
    upload(app, "syn-methods")
    FakeModels.reply = ('The encoder has 6 layers.\n\nEvidence:\n'
                        '- [Page 2] "The passage encoder uses 6 transformer layers"\n'
                        '- [Page 1] "GraphLens reaches 91.2% accuracy on CiteNet"\n'
                        '- [Page 2] "a sentence the paper never contains at all"')
    app.chat_input[0].set_value("How many layers?").run()
    assert not app.exception
    assert "[Page 2]" in FakeModels.last_prompt and "not instructions" in FakeModels.last_prompt
    message = app.session_state.chat_messages[-1]
    assert message["content"] == "The encoder has 6 layers."
    assert [c["status"] for c in message["evidence"]] == ["verified", "wrong_page", "not_found"]
    labels = [e.label for e in app.expander]
    assert any(label.startswith("Sources: 2 of 3") for label in labels)


def test_abstention_is_recorded(app):
    upload(app, "syn-methods")
    FakeModels.reply = "Not stated in the paper. It reports accuracy, not cost.\n\nEvidence:"
    app.chat_input[0].set_value("What did training cost?").run()
    message = app.session_state.chat_messages[-1]
    assert message["abstained"] and message["evidence"] == []


def test_exports_include_evidence_and_extraction_notes(app):
    upload(app, "syn-scanned")
    FakeModels.reply = 'Agreement is 0.81.\n\nEvidence:\n- [Page 3] "Cohen\'s kappa is 0.81"'
    app.chat_input[0].set_value("What is the agreement?").run()
    report = next(m.value for m in app.markdown if m.value.startswith("# GraphLens"))
    assert "Found on page 3" in report and "## Extraction Notes" in report and "scanned" in report


def test_source_text_viewer_shows_pages(app):
    upload(app, "syn-methods")
    assert any(e.label == "Source text" for e in app.expander)
    app.number_input[0].set_value(2).run()
    assert any("passage encoder uses 6 transformer layers" in t.value for t in app.text)
