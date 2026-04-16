from __future__ import annotations

from io import BytesIO

import pytest
from docx import Document

from app.services import text_extractor
from app.services.text_extractor import (
    FileTooLarge,
    UnsupportedFileType,
    extract_pages,
    extract_text,
)


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    document = Document()
    for p in paragraphs:
        document.add_paragraph(p)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_rejects_unknown_extension() -> None:
    with pytest.raises(UnsupportedFileType):
        extract_text("resume.zip", b"garbage")


def test_rejects_missing_extension() -> None:
    with pytest.raises(UnsupportedFileType):
        extract_text("resume", b"hello")


def test_enforces_size_limit() -> None:
    with pytest.raises(FileTooLarge):
        extract_text("resume.txt", b"x" * 1024, max_bytes=100)


def test_txt_extraction_and_whitespace_normalization() -> None:
    raw = b"Name: Anu\r\n\r\n\r\n\r\nSkills:   Python,   FastAPI   \n"
    out = extract_text("resume.txt", raw)
    assert "Name: Anu" in out
    assert "Skills: Python, FastAPI" in out
    assert "\n\n\n" not in out


def test_docx_extraction() -> None:
    data = _make_docx_bytes(["Jane Doe", "Senior Backend Engineer", "Skills: Python, FastAPI"])
    out = extract_text("jane.docx", data)
    assert "Jane Doe" in out
    assert "FastAPI" in out


def test_extract_pages_pdf_returns_one_entry_per_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakePdf:
        pages = [FakePage("Page one text"), FakePage("Page two text")]

        def __enter__(self) -> "FakePdf":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(text_extractor.pdfplumber, "open", lambda _s: FakePdf())

    pages = extract_pages("resume.pdf", b"%PDF-fake")
    assert pages == ["Page one text", "Page two text"]
    assert extract_text("resume.pdf", b"%PDF-fake") == "Page one text\n\nPage two text"


def test_extract_pages_txt_paragraph_packs() -> None:
    body = ("para " + "x" * 2000 + "\n\n") * 3
    pages = extract_pages("resume.txt", body.encode())
    assert len(pages) >= 2
    assert all(len(p) <= 3500 for p in pages)
    assert "".join(pages).count("x" * 2000) == 3


def test_pdf_dispatch_invokes_pdfplumber(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return "Hello from PDF"

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self) -> "FakePdf":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_open(_stream: object) -> FakePdf:
        return FakePdf()

    monkeypatch.setattr(text_extractor.pdfplumber, "open", fake_open)
    assert extract_text("resume.pdf", b"%PDF-fake") == "Hello from PDF"
