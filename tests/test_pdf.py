import pymupdf
import pytest

from core.pdf import ExtractionError, extract_pdf, extraction_warnings

BODY = "This page has selectable text describing the experimental setup in detail. " * 3


def make_pdf(pages=2, scanned=(), tables=0):
    doc = pymupdf.open()
    for n in range(1, pages + 1):
        page = doc.new_page()
        if n in scanned:
            src = pymupdf.open(); s = src.new_page(); s.insert_textbox(pymupdf.Rect(50, 50, 550, 300), f"Page {n} {BODY}")
            page.insert_image(page.rect, pixmap=s.get_pixmap(dpi=72))  # image only, no text layer
        else:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 300), f"Page {n} {BODY}", fontsize=10)
        for t in range(tables):
            top, left, width, height = 350 + t * 130, 60, 150, 25
            for r in range(5):
                page.draw_line((left, top + r * height), (left + 3 * width, top + r * height))
            for c in range(4):
                page.draw_line((left + c * width, top), (left + c * width, top + 4 * height))
            for r in range(4):
                for c in range(3):
                    cell = pymupdf.Rect(left + c * width, top + r * height, left + (c + 1) * width, top + (r + 1) * height)
                    page.insert_textbox(cell + (4, 4, -4, -4), f"r{r}c{c}", fontsize=9)
    return doc.tobytes()


def test_page_boundaries_kept_and_complete():
    result = extract_pdf(make_pdf(3))
    assert len(result["pages"]) == 3 and "Page 2" in result["pages"][1]
    assert extraction_warnings(result["report"]) == []


@pytest.mark.parametrize("data", [b"", b"not a pdf at all"])
def test_malformed_input_raises(data):
    with pytest.raises(ExtractionError):
        extract_pdf(data)


def test_truncated_pdf_is_flagged_not_silent():
    data = make_pdf(2)
    try:
        result = extract_pdf(data[: len(data) // 2])
    except ExtractionError:
        return  # refusing a damaged file is also acceptable
    assert result["report"]["repaired"]
    assert any("damaged" in w for w in extraction_warnings(result["report"]))


def test_user_password_pdf_refused_owner_only_allowed():
    doc = pymupdf.open(stream=make_pdf(1), filetype="pdf")
    locked = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    with pytest.raises(ExtractionError, match="password"):
        extract_pdf(locked)
    owner_only = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", permissions=pymupdf.PDF_PERM_PRINT)
    assert "Page 1" in extract_pdf(owner_only)["pages"][0]


def test_scanned_pages_are_reported():
    result = extract_pdf(make_pdf(3, scanned={2}))
    assert result["report"]["pages_without_text"] == [2]
    assert any("scanned" in w and "2" in w for w in extraction_warnings(result["report"]))


def test_page_limit_is_reported():
    result = extract_pdf(make_pdf(5), max_pages=2)
    assert len(result["pages"]) == 2
    assert any("first 2 of 5" in w for w in extraction_warnings(result["report"]))


def test_table_heavy_pdf():
    result = extract_pdf(make_pdf(2, tables=2))
    assert len(result["tables"]) == 4
    assert result["tables"][0]["rows"][0] == ["r0c0", "r0c1", "r0c2"]
