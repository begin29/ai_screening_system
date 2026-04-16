from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import database
from app.main import app
from app.models import JobTemplate, Recommendation
from app.schemas import ScreeningResult
from app.services import screener


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
                matching_skills=["Python"],
                missing_skills=[],
                experience="5",
                recommendation="Good fit" if score >= 70 else "Bad fit",
            ),
            "{}",
        )

    return _inner


def test_index_renders_with_empty_db(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_index_lists_existing_templates(client: TestClient) -> None:
    with database.SessionLocal() as db:
        db.add(JobTemplate(name="Backend Engineer", content="Python + FastAPI"))
        db.commit()

    response = client.get("/")
    assert response.status_code == 200
    assert "Backend Engineer" in response.text


def test_screen_form_renders_results(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        screener.llm_client, "screen_one", _mock_screen_one("Anu", 85)
    )
    files = [("resumes", ("anu.txt", b"Anu resume", "text/plain"))]
    response = client.post(
        "/screen",
        data={"job_description": "Python backend"},
        files=files,
    )
    assert response.status_code == 200
    assert "Anu" in response.text


def test_screen_form_reshows_index_on_unsupported_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screener.llm_client, "screen_one", _mock_screen_one("X", 50))
    files = [("resumes", ("resume.zip", b"junk", "application/zip"))]
    response = client.post(
        "/screen",
        data={"job_description": "JD"},
        files=files,
    )
    assert response.status_code == 400
    assert ".zip" in response.text


def test_create_template_redirects_to_home(client: TestClient) -> None:
    response = client.post(
        "/templates",
        data={"name": "Data Engineer", "content": "SQL + Python"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"

    with database.SessionLocal() as db:
        tpl = db.query(JobTemplate).filter_by(name="Data Engineer").one()
        assert tpl.content == "SQL + Python"


def test_create_template_rejects_duplicate_name(client: TestClient) -> None:
    with database.SessionLocal() as db:
        db.add(JobTemplate(name="Dup", content="first"))
        db.commit()

    response = client.post(
        "/templates",
        data={"name": "Dup", "content": "second"},
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert "already exists" in response.text


def test_delete_template_removes_row(client: TestClient) -> None:
    with database.SessionLocal() as db:
        tpl = JobTemplate(name="Temp", content="x")
        db.add(tpl)
        db.commit()
        template_id = tpl.id

    response = client.post(
        f"/templates/{template_id}/delete", follow_redirects=False
    )
    assert response.status_code == 303

    with database.SessionLocal() as db:
        assert db.get(JobTemplate, template_id) is None


def test_delete_template_is_idempotent_for_missing_id(client: TestClient) -> None:
    response = client.post("/templates/9999/delete", follow_redirects=False)
    assert response.status_code == 303


def test_view_scan_returns_404_for_unknown_id(client: TestClient) -> None:
    response = client.get("/scans/12345")
    assert response.status_code == 404


def test_view_scan_renders_stored_results(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        screener.llm_client, "screen_one", _mock_screen_one("Stored", 90)
    )
    files = [("resumes", ("s.txt", b"stored resume", "text/plain"))]
    create = client.post(
        "/screen",
        data={"job_description": "Stored JD"},
        files=files,
    )
    assert create.status_code == 200

    with database.SessionLocal() as db:
        from app.models import JobDescription, Screening

        jd = db.query(JobDescription).order_by(JobDescription.id.desc()).first()
        assert jd is not None
        screenings = db.query(Screening).filter_by(job_description_id=jd.id).all()
        assert len(screenings) == 1
        assert screenings[0].recommendation is Recommendation.GOOD_FIT

    response = client.get(f"/scans/{jd.id}")
    assert response.status_code == 200
    assert "Stored" in response.text
    assert "Stored JD" in response.text
