from __future__ import annotations

import re

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def pack_chunks(pages: list[str], max_chars: int) -> list[str]:
    """Greedy-pack ordered pages into chunks <= ``max_chars`` characters.

    A single page larger than ``max_chars`` is split on paragraph boundaries
    (and, as a last resort, on hard character boundaries) so that every
    returned chunk respects the limit.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for page in pages:
        page = page.strip()
        if not page:
            continue

        if len(page) > max_chars:
            flush()
            chunks.extend(_split_oversized(page, max_chars))
            continue

        added_len = len(page) + (2 if current else 0)
        if current and current_len + added_len > max_chars:
            flush()
            added_len = len(page)
        current.append(page)
        current_len += added_len

    flush()
    return chunks


def _split_oversized(text: str, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    out: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                out.append("\n\n".join(current))
                current = []
                current_len = 0
            for i in range(0, len(para), max_chars):
                out.append(para[i : i + max_chars])
            continue
        added_len = len(para) + (2 if current else 0)
        if current and current_len + added_len > max_chars:
            out.append("\n\n".join(current))
            current = []
            current_len = 0
            added_len = len(para)
        current.append(para)
        current_len += added_len
    if current:
        out.append("\n\n".join(current))
    return out
