from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ScreeningResult
from app.services import llm_client, screener


@pytest.fixture
def client(tmp_db: None) -> TestClient:
    return TestClient(app)


def _mock_screen_one(name: str, score: int):
    async def _inner(
        _jd: str, _text: str, **_kwargs
    ) -> tuple[ScreeningResult, str]:
        return (
            ScreeningResult(
                candidate_name=name,
                match_score=score,
                matching_skills=["Python", "FastAPI"],
                missing_skills=["Docker"],
                experience="5",
                recommendation="Good fit" if score >= 70 else "Bad fit",
            ),
            '{"ok": true}',
        )

    return _inner


def test_health_reports_llm_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_reachable() -> bool:
        return False

    monkeypatch.setattr(llm_client, "check_reachable", fake_reachable)

    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "llm_reachable": False}


def test_screen_returns_spec_shape(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screener.llm_client, "screen_one", _mock_screen_one("Anu", 85))

    files = [("resumes", ("anu.txt", b"Anu, Python, FastAPI", "text/plain"))]
    response = client.post(
        "/api/screen",
        data={"job_description": "Looking for a Python backend engineer"},
        files=files,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job_description_id"] >= 1
    assert len(body["results"]) == 1

    result = body["results"][0]
    assert set(result.keys()) == {
        "candidate_name",
        "match_score",
        "matching_skills",
        "missing_skills",
        "experience",
        "recommendation",
    }
    assert result == {
        "candidate_name": "Anu",
        "match_score": 85,
        "matching_skills": ["Python", "FastAPI"],
        "missing_skills": ["Docker"],
        "experience": "5",
        "recommendation": "Good fit",
    }


def test_screen_ranks_multiple_resumes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def screen_one(
        _jd: str, resume_text: str, **_kwargs
    ) -> tuple[ScreeningResult, str]:
        if b"alice" in resume_text.lower().encode():
            score, name = 90, "Alice"
        elif b"bob" in resume_text.lower().encode():
            score, name = 40, "Bob"
        else:
            score, name = 70, "Unknown"
        return (
            ScreeningResult(
                candidate_name=name,
                match_score=score,
                matching_skills=[],
                missing_skills=[],
                experience="3",
                recommendation="Good fit" if score >= 70 else "Bad fit",
            ),
            "{}",
        )

    monkeypatch.setattr(screener.llm_client, "screen_one", screen_one)

    files = [
        ("resumes", ("alice.txt", b"alice resume text", "text/plain")),
        ("resumes", ("bob.txt", b"bob resume text", "text/plain")),
    ]
    response = client.post(
        "/api/screen",
        data={"job_description": "Python role"},
        files=files,
    )
    assert response.status_code == 200
    names = [r["candidate_name"] for r in response.json()["results"]]
    assert names == ["Alice", "Bob"]


def test_screen_rejects_unsupported_extension(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screener.llm_client, "screen_one", _mock_screen_one("X", 50))

    files = [("resumes", ("resume.zip", b"not a resume", "application/zip"))]
    response = client.post(
        "/api/screen",
        data={"job_description": "JD"},
        files=files,
    )
    assert response.status_code == 400
    assert ".zip" in response.json()["detail"]
