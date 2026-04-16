from __future__ import annotations

import pytest

from app.services.chunker import pack_chunks


def test_packs_small_pages_into_single_chunk() -> None:
    pages = ["alpha", "beta", "gamma"]
    chunks = pack_chunks(pages, max_chars=1000)
    assert chunks == ["alpha\n\nbeta\n\ngamma"]


def test_starts_new_chunk_when_budget_exceeded() -> None:
    pages = ["a" * 1000, "b" * 1000, "c" * 1000]
    chunks = pack_chunks(pages, max_chars=2500)
    assert len(chunks) == 2
    assert chunks[0].startswith("a" * 1000)
    assert "b" * 1000 in chunks[0]
    assert chunks[1] == "c" * 1000


def test_splits_oversized_page_on_paragraphs() -> None:
    page = "para one.\n\n" + "x" * 200 + "\n\npara three."
    chunks = pack_chunks([page], max_chars=100)
    assert len(chunks) >= 2
    joined = "\n".join(chunks)
    assert "para one." in joined
    assert "para three." in joined
    assert all(len(c) <= 200 for c in chunks)


def test_hard_splits_single_paragraph_larger_than_max() -> None:
    page = "y" * 250
    chunks = pack_chunks([page], max_chars=100)
    assert len(chunks) == 3
    assert chunks[0] == "y" * 100
    assert chunks[2] == "y" * 50


def test_empty_and_whitespace_pages_skipped() -> None:
    chunks = pack_chunks(["", "   ", "real content"], max_chars=100)
    assert chunks == ["real content"]


def test_rejects_nonpositive_max_chars() -> None:
    with pytest.raises(ValueError):
        pack_chunks(["x"], max_chars=0)
