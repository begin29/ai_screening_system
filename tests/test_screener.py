from __future__ import annotations

import pytest

from app import database
from app.schemas import ScreeningResult
from app.services import screener
from app.services.screener import ResumeUpload, screen_resumes


def _result(name: str, score: int) -> ScreeningResult:
    return ScreeningResult(
        candidate_name=name,
        match_score=score,
        matching_skills=["Python"],
        missing_skills=["Docker"],
        experience="5",
        recommendation="Good fit" if score >= 70 else "Bad fit",
    )


@pytest.mark.asyncio
async def test_screens_and_ranks_by_score(
    monkeypatch: pytest.MonkeyPatch, tmp_db: None
) -> None:
    canned = {
        "alice.txt": _result("Alice", 90),
        "bob.txt": _result("Bob", 55),
        "carol.txt": _result("Carol", 72),
    }

    async def fake_screen_one(
        _jd: str, resume_text: str, **_kwargs
    ) -> tuple[ScreeningResult, str]:
        for name, res in canned.items():
            if name.split(".")[0].lower() in resume_text.lower():
                return res, '{"ok": true}'
        raise AssertionError(f"unexpected resume_text: {resume_text!r}")

    monkeypatch.setattr(screener.llm_client, "screen_one", fake_screen_one)

    uploads = [
        ResumeUpload(filename=name, data=name.split(".")[0].encode())
        for name in canned
    ]

    with database.SessionLocal() as db:
        response = await screen_resumes("Python job", uploads, db=db)

    scores = [r.match_score for r in response.results]
    assert scores == sorted(scores, reverse=True)
    assert [r.candidate_name for r in response.results] == ["Alice", "Carol", "Bob"]
    assert response.results[-1].recommendation == "Bad fit"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_db: None
) -> None:
    async def failing(
        _jd: str, _text: str, **_kwargs
    ) -> tuple[ScreeningResult, str]:
        raise RuntimeError("LLM backend is down")

    monkeypatch.setattr(screener.llm_client, "screen_one", failing)

    uploads = [ResumeUpload(filename="anu.txt", data=b"Anu resume")]
    with database.SessionLocal() as db:
        response = await screen_resumes("JD", uploads, db=db)

    assert len(response.results) == 1
    assert response.results[0].match_score == 0
    assert response.results[0].recommendation == "Bad fit"
