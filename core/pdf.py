"""PDF extraction that keeps page boundaries and records what could not be extracted."""
import re

import pymupdf

MIN_FIGURE_SIZE = 80  # px; smaller images are usually logos or icons
MAX_FIGURES = 60
MIN_PAGE_CHARS = 20  # below this, a page is treated as having no text layer (scanned or blank)


class ExtractionError(ValueError):
    pass


def extract_pdf(pdf_bytes, max_pages=300):
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except pymupdf.FileDataError as e:
        raise ExtractionError(f"The file could not be read as a PDF ({e}).") from e

    with doc:
        if doc.needs_pass:
            raise ExtractionError("This PDF is password-protected. Please upload an unlocked copy.")

        report = {
            "page_count": doc.page_count,
            "pages_processed": min(doc.page_count, max_pages),
            "pages_without_text": [],
            "repaired": bool(doc.is_repaired),
            "images_not_examined": 0,
            "table_errors": [],
        }

        metadata = {}
        pdf_meta = doc.metadata or {}
        if pdf_meta.get("title", "").strip():
            metadata["title"] = pdf_meta["title"].strip()
        if pdf_meta.get("author", "").strip():
            metadata["authors"] = [a.strip() for a in re.split(r";|\band\b", pdf_meta["author"]) if a.strip()]

        pages, figures, tables, seen_xrefs = [], [], [], set()
        for number in range(1, report["pages_processed"] + 1):
            page = doc[number - 1]
            text = page.get_text()
            pages.append(text)
            if len(text.strip()) < MIN_PAGE_CHARS:
                report["pages_without_text"].append(number)

            for image in page.get_images(full=True):
                xref = image[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                if len(figures) >= MAX_FIGURES:
                    report["images_not_examined"] += 1
                    continue
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.width < MIN_FIGURE_SIZE or pix.height < MIN_FIGURE_SIZE:
                        continue
                    if pix.n - pix.alpha >= 4:  # CMYK and similar -> RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    figures.append({"page": number, "data": pix.tobytes("png"),
                                    "width": pix.width, "height": pix.height})
                except Exception:
                    continue  # skip images in formats PyMuPDF can't convert

            try:
                found = page.find_tables()
            except Exception:  # table detection is best-effort, but record that it failed
                report["table_errors"].append(number)
                continue
            for table in found.tables:
                # table.extract() misplaces decimal points in some papers ("67.0%" -> "670%\n."); reading each
                # cell's box from the page keeps the characters in reading order
                rows = [[" ".join(page.get_textbox(cell).split()) if cell else "" for cell in row.cells]
                        for row in table.rows]
                rows = _drop_empty_columns(rows)
                if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    tables.append({"page": number, "rows": rows})

    return {"pages": pages, "metadata": metadata, "figures": figures, "tables": tables, "report": report}


def _drop_empty_columns(rows):
    # PyMuPDF often reports blank spacer columns between real ones
    width = max((len(r) for r in rows), default=0)
    keep = [c for c in range(width) if any(c < len(r) and r[c].strip() for r in rows)]
    return [[r[c] if c < len(r) else "" for c in keep] for r in rows]


def extraction_warnings(report):
    """Plain-language notes on anything the extraction missed; empty when it looks complete."""
    notes = []
    if report["pages_processed"] < report["page_count"]:
        notes.append(f"Only the first {report['pages_processed']} of {report['page_count']} pages were processed.")
    missing = report["pages_without_text"]
    if missing:
        shown = ", ".join(map(str, missing[:10])) + ("…" if len(missing) > 10 else "")
        notes.append(f"{len(missing)} page(s) have no selectable text and were not read (likely scanned): {shown}.")
    if report["repaired"]:
        notes.append("The PDF was damaged and had to be repaired; some content may be missing.")
    if report["images_not_examined"]:
        notes.append(f"{report['images_not_examined']} more images were not extracted (limit {MAX_FIGURES}).")
    if report["table_errors"]:
        notes.append("Table detection failed on page(s) " + ", ".join(map(str, report["table_errors"])) + ".")
    return notes
