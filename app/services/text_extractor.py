from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import pdfplumber
from docx import Document

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".doc", ".docx", ".txt"})

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RE = re.compile(r"\n{3,}")
_CID_GLYPH_RE = re.compile(r"\(cid:\d+\)")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

PSEUDO_PAGE_TARGET_CHARS = 3000


class UnsupportedFileType(ValueError):
    """Raised when a file extension is not in the allow-list."""


class FileTooLarge(ValueError):
    """Raised when a file exceeds the configured size limit."""


class ExtractionError(RuntimeError):
    """Raised when a supported file cannot be parsed (e.g. corrupt PDF,
    missing libreoffice for .doc)."""


def extract_pages(
    filename: str, data: bytes, max_bytes: int | None = None
) -> list[str]:
    """Return resume content as a list of page-sized strings.

    PDFs are split on real page boundaries via pdfplumber; the other formats
    have no page concept and are paragraph-packed into pseudo-pages of
    roughly ``PSEUDO_PAGE_TARGET_CHARS`` characters.
    """
    if max_bytes is not None and len(data) > max_bytes:
        raise FileTooLarge(
            f"{filename} is {len(data)} bytes, exceeds limit of {max_bytes} bytes"
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        pretty = suffix or "(none)"
        raise UnsupportedFileType(f"Unsupported file type: {pretty}")

    if suffix == ".pdf":
        raw_pages = _extract_pdf_pages(data)
    elif suffix == ".docx":
        raw_pages = _paragraph_pack(_extract_docx_raw(data))
    elif suffix == ".doc":
        raw_pages = _paragraph_pack(_extract_doc_raw(data))
    else:
        raw_pages = _paragraph_pack(_extract_txt_raw(data))

    normalized = [_normalize(page) for page in raw_pages]
    return [page for page in normalized if page]


def extract_text(filename: str, data: bytes, max_bytes: int | None = None) -> str:
    pages = extract_pages(filename, data, max_bytes=max_bytes)
    return _normalize("\n\n".join(pages))


def _extract_pdf_pages(data: bytes) -> list[str]:
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            return [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise ExtractionError(f"Failed to parse PDF: {exc}") from exc


def _extract_docx_raw(data: bytes) -> str:
    try:
        document = Document(BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Failed to parse DOCX: {exc}") from exc
    return "\n".join(p.text for p in document.paragraphs)


def _extract_doc_raw(data: bytes) -> str:
    binary = shutil.which("libreoffice") or shutil.which("soffice")
    if binary is None:
        raise ExtractionError(
            "Parsing .doc files requires LibreOffice. Install it (e.g. "
            "`brew install --cask libreoffice` on macOS) or upload the "
            "resume as .pdf, .docx, or .txt."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "input.doc"
        src.write_bytes(data)

        try:
            subprocess.run(
                [
                    binary,
                    "--headless",
                    "--convert-to",
                    "txt:Text (encoded):UTF8",
                    "--outdir",
                    str(tmp_path),
                    str(src),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExtractionError("LibreOffice timed out converting .doc file") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise ExtractionError(
                f"LibreOffice failed to convert .doc file: {stderr}"
            ) from exc

        out = tmp_path / "input.txt"
        if not out.exists():
            raise ExtractionError("LibreOffice produced no output for .doc file")
        return out.read_text(encoding="utf-8", errors="replace")


def _extract_txt_raw(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _paragraph_pack(
    text: str, target_chars: int = PSEUDO_PAGE_TARGET_CHARS
) -> list[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    if not paragraphs:
        return []

    pages: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        if current and current_len + para_len > target_chars:
            pages.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len + 2
    if current:
        pages.append("\n\n".join(current))
    return pages


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CID_GLYPH_RE.sub("-", text)
    lines = [_WHITESPACE_RE.sub(" ", line).rstrip() for line in text.split("\n")]
    collapsed = "\n".join(lines)
    return _BLANK_LINE_RE.sub("\n\n", collapsed).strip()
