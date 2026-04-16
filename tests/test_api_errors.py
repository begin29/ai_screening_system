from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import api
from app.services.text_extractor import ExtractionError, FileTooLarge


@pytest.fixture
def client(tmp_db: None) -> TestClient:
    return TestClient(app)


def test_screen_returns_413_when_file_too_large(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_screen(**_kwargs: object):
        raise FileTooLarge("resume.pdf is 50 bytes, exceeds limit of 10 bytes")

    monkeypatch.setattr(api, "screen_resumes", fake_screen)

    files = [("resumes", ("resume.pdf", b"x" * 50, "application/pdf"))]
    response = client.post(
        "/api/screen",
        data={"job_description": "Python role"},
        files=files,
    )

    assert response.status_code == 413
    assert "exceeds limit" in response.json()["detail"]


def test_screen_returns_422_when_extraction_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_screen(**_kwargs: object):
        raise ExtractionError("Failed to parse PDF: corrupt stream")

    monkeypatch.setattr(api, "screen_resumes", fake_screen)

    files = [("resumes", ("broken.pdf", b"%PDF-corrupt", "application/pdf"))]
    response = client.post(
        "/api/screen",
        data={"job_description": "Python role"},
        files=files,
    )

    assert response.status_code == 422
    assert "Failed to parse PDF" in response.json()["detail"]


def test_screen_rejects_empty_job_description(client: TestClient) -> None:
    files = [("resumes", ("r.txt", b"hello", "text/plain"))]
    response = client.post(
        "/api/screen",
        data={"job_description": ""},
        files=files,
    )
    assert response.status_code == 422
