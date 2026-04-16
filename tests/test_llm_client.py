from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app import config
from app.schemas import ChunkDigest, RoleEntry, ScreeningResult
from app.services import llm_client
from app.services.llm_client import (
    LLMError,
    _align_recommendation,
    _format_digest,
    _parse_json,
    screen_one,
)


def _result(score: int, recommendation: str) -> ScreeningResult:
    return ScreeningResult(
        candidate_name="Anu",
        match_score=score,
        matching_skills=["Python"],
        missing_skills=[],
        experience="5",
        recommendation=recommendation,
    )


class TestParseJson:
    def test_parses_valid_json_directly(self) -> None:
        assert _parse_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}

    def test_extracts_json_wrapped_in_prose(self) -> None:
        raw = 'Here is the result:\n{"candidate_name": "Anu", "match_score": 80}\nThanks.'
        assert _parse_json(raw) == {"candidate_name": "Anu", "match_score": 80}

    def test_extracts_json_from_markdown_fence(self) -> None:
        raw = '```json\n{"match_score": 55}\n```'
        assert _parse_json(raw) == {"match_score": 55}

    def test_raises_when_no_json_object_present(self) -> None:
        with pytest.raises(LLMError, match="no JSON object"):
            _parse_json("the model refused to answer")

    def test_raises_when_extracted_json_is_malformed(self) -> None:
        with pytest.raises(LLMError, match="not valid JSON"):
            _parse_json("prefix {not: valid, json: here} suffix")


class TestAlignRecommendation:
    def test_upgrades_bad_fit_when_score_at_threshold(self) -> None:
        aligned = _align_recommendation(_result(70, "Bad fit"))
        assert aligned.recommendation == "Good fit"

    def test_upgrades_bad_fit_when_score_above_threshold(self) -> None:
        aligned = _align_recommendation(_result(95, "Bad fit"))
        assert aligned.recommendation == "Good fit"

    def test_downgrades_good_fit_when_score_below_threshold(self) -> None:
        aligned = _align_recommendation(_result(40, "Good fit"))
        assert aligned.recommendation == "Bad fit"

    def test_keeps_consistent_recommendation_unchanged(self) -> None:
        original = _result(85, "Good fit")
        aligned = _align_recommendation(original)
        assert aligned.recommendation == "Good fit"

    def test_returns_same_instance_when_already_aligned(self) -> None:
        original = _result(30, "Bad fit")
        aligned = _align_recommendation(original)
        assert aligned is original


class _FakeCompletions:
    def __init__(
        self, screening_response: dict, digest_responses: list[dict]
    ) -> None:
        self.screening_response = screening_response
        self.digest_responses = list(digest_responses)
        self.calls: list[dict] = []

    async def create(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        response_format: dict,
        **_kwargs: object,
    ):
        schema_name = response_format["json_schema"]["name"]
        self.calls.append(
            {"schema": schema_name, "content": messages[0]["content"]}
        )
        payload = (
            self.digest_responses.pop(0)
            if schema_name == "chunk_digest"
            else self.screening_response
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )
            ]
        )


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    screening_response: dict,
    digest_responses: list[dict] | None = None,
) -> _FakeCompletions:
    fake = _FakeCompletions(screening_response, digest_responses or [])
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    llm_client.get_client.cache_clear()
    monkeypatch.setattr(llm_client, "get_client", lambda: client)
    return fake


def _screening_payload(**overrides) -> dict:
    base = {
        "notes": "ok",
        "candidate_name": "Anu",
        "match_score": 80,
        "matching_skills": ["Python"],
        "missing_skills": [],
        "experience": "10",
        "recommendation": "Good fit",
    }
    base.update(overrides)
    return base


def _digest_payload(**overrides) -> dict:
    base = {
        "candidate_name": "",
        "stated_years_experience": 0,
        "skills_found": [],
        "role_entries": [],
        "highlights": "",
    }
    base.update(overrides)
    return base


@pytest.fixture
def small_context(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_MAX_RESUME_CHARS", "200")
    monkeypatch.setenv("LLM_MAX_CHUNK_CHARS", "150")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.mark.asyncio
async def test_short_path_issues_single_screening_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake(
        monkeypatch,
        screening_response=_screening_payload(
            candidate_name="Anu", matching_skills=["Python"]
        ),
    )
    resume = "Anu, Python, 10 years of experience"

    result, _ = await screen_one("Python role", resume)

    assert [c["schema"] for c in fake.calls] == ["screening_result"]
    assert result.candidate_name == "Anu"
    assert result.matching_skills == ["Python"]


@pytest.mark.asyncio
async def test_long_path_extracts_each_chunk_then_scores(
    monkeypatch: pytest.MonkeyPatch, small_context: None
) -> None:
    digest_1 = _digest_payload(
        candidate_name="Anu",
        stated_years_experience=12,
        skills_found=["Python", "FastAPI"],
        highlights="Profile and summary.",
    )
    digest_2 = _digest_payload(
        skills_found=["python", "PostgreSQL"],
        role_entries=[
            {"title": "Senior", "company": "Acme", "start": "2018", "end": "2024"}
        ],
        highlights="Experience section.",
    )
    fake = _install_fake(
        monkeypatch,
        screening_response=_screening_payload(
            matching_skills=["Python", "PostgreSQL"],
            missing_skills=["Docker"],
            match_score=82,
        ),
        digest_responses=[digest_1, digest_2],
    )

    pages = [
        "Profile: Anu, 12 years with Python and FastAPI. " * 3,
        "Experience: Senior at Acme, PostgreSQL heavy. " * 3,
    ]
    full_resume = "\n\n".join(pages)
    assert len(full_resume) > 200

    result, _ = await screen_one("Senior Python role", full_resume, pages=pages)

    schemas = [c["schema"] for c in fake.calls]
    assert schemas.count("chunk_digest") == 2
    assert schemas.count("screening_result") == 1

    final_call = next(c for c in fake.calls if c["schema"] == "screening_result")
    content = final_call["content"]
    assert "Skills observed:" in content
    assert "Python" in content
    assert "PostgreSQL" in content
    assert "Stated years of experience: 12" in content
    assert "Senior @ Acme" in content

    assert result.match_score == 82
    assert "Python" in result.matching_skills
    assert "PostgreSQL" in result.matching_skills


@pytest.mark.asyncio
async def test_long_path_fabrication_filter_uses_full_resume(
    monkeypatch: pytest.MonkeyPatch, small_context: None
) -> None:
    _install_fake(
        monkeypatch,
        screening_response=_screening_payload(
            matching_skills=["Python", "MadeUpSkill"],
            missing_skills=[],
        ),
        digest_responses=[_digest_payload(), _digest_payload()],
    )

    pages = [
        "this resume mentions Python and other things. " * 3,
        "nothing else of particular note on this page. " * 3,
    ]
    full_resume = "\n\n".join(pages)
    assert len(full_resume) > 200

    result, _ = await screen_one("JD", full_resume, pages=pages)

    assert "Python" in result.matching_skills
    assert "MadeUpSkill" not in result.matching_skills
    assert "MadeUpSkill" in result.missing_skills


class TestFormatDigest:
    def test_dedupes_skills_case_insensitively(self) -> None:
        digests = [
            ChunkDigest(skills_found=["Python", "FastAPI"]),
            ChunkDigest(skills_found=["python", "PostgreSQL", "fastapi"]),
        ]
        blob = _format_digest(digests)
        skills_line = next(
            line for line in blob.splitlines() if line.startswith("Skills observed:")
        )
        skills = [
            s.strip()
            for s in skills_line.removeprefix("Skills observed:").split(",")
        ]
        assert skills == ["Python", "FastAPI", "PostgreSQL"]

    def test_picks_first_non_empty_name_and_max_years(self) -> None:
        digests = [
            ChunkDigest(candidate_name="", stated_years_experience=5),
            ChunkDigest(candidate_name="Anu", stated_years_experience=12),
            ChunkDigest(candidate_name="Someone Else", stated_years_experience=3),
        ]
        blob = _format_digest(digests)
        assert "Name: Anu" in blob
        assert "Stated years of experience: 12" in blob

    def test_dedupes_role_entries(self) -> None:
        role = RoleEntry(title="Senior", company="Acme", start="2018", end="2024")
        digests = [
            ChunkDigest(role_entries=[role]),
            ChunkDigest(role_entries=[role]),
        ]
        blob = _format_digest(digests)
        assert blob.count("Senior @ Acme") == 1

    def test_includes_chunk_markers_in_highlights(self) -> None:
        digests = [
            ChunkDigest(highlights="first"),
            ChunkDigest(highlights="second"),
        ]
        blob = _format_digest(digests)
        assert "--- chunk 1 --- first" in blob
        assert "--- chunk 2 --- second" in blob
