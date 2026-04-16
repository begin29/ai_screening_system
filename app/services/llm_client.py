from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import lru_cache

import httpx
from openai import AsyncOpenAI
from openai import APIError, APIConnectionError, APITimeoutError
from pydantic import ValidationError

from app.config import get_settings
from app.prompts.extraction import (
    CHUNK_DIGEST_JSON_SCHEMA,
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)
from app.prompts.screening import SYSTEM_PROMPT, build_user_prompt
from app.schemas import ChunkDigest, RoleEntry, ScreeningResult
from app.services.chunker import pack_chunks

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_llm_semaphore: asyncio.Semaphore | None = None
_llm_semaphore_size: int = 0


def _get_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore, _llm_semaphore_size
    desired = max(1, get_settings().llm_max_concurrency)
    if _llm_semaphore is None or _llm_semaphore_size != desired:
        _llm_semaphore = asyncio.Semaphore(desired)
        _llm_semaphore_size = desired
    return _llm_semaphore

SCREENING_JSON_SCHEMA: dict = {
    "name": "screening_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "notes": {
                "type": "string",
                "description": (
                    "Short reasoning (2-4 sentences) about how the resume "
                    "matches the job description. Written BEFORE deriving "
                    "the other fields."
                ),
            },
            "candidate_name": {"type": "string"},
            "match_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "matching_skills": {"type": "array", "items": {"type": "string"}},
            "missing_skills": {"type": "array", "items": {"type": "string"}},
            "experience": {"type": "string"},
            "recommendation": {
                "type": "string",
                "enum": ["Good fit", "Bad fit"],
            },
        },
        "required": [
            "notes",
            "candidate_name",
            "match_score",
            "matching_skills",
            "missing_skills",
            "experience",
            "recommendation",
        ],
        "additionalProperties": False,
    },
}


class LLMError(RuntimeError):
    """Raised when the LLM call fails or returns unusable content."""


@lru_cache
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
    )


async def screen_one(
    job_description: str,
    resume_text: str,
    *,
    pages: list[str] | None = None,
) -> tuple[ScreeningResult, str]:
    settings = get_settings()
    if len(resume_text) <= settings.llm_max_resume_chars:
        return await _screen_short(job_description, resume_text)
    return await _screen_long(
        job_description, pages or [resume_text], resume_text
    )


async def _screen_short(
    job_description: str, resume_text: str
) -> tuple[ScreeningResult, str]:
    settings = get_settings()
    trimmed_jd = _truncate(job_description, settings.llm_max_jd_chars)
    trimmed_resume = _truncate(resume_text, settings.llm_max_resume_chars)
    user_content = (
        SYSTEM_PROMPT
        + "\n\n"
        + build_user_prompt(trimmed_jd, trimmed_resume)
    )
    return await _run_screening(user_content, haystack=trimmed_resume)


async def _screen_long(
    job_description: str,
    pages: list[str],
    full_resume_text: str,
) -> tuple[ScreeningResult, str]:
    settings = get_settings()
    chunks = pack_chunks(pages, settings.llm_max_chunk_chars)
    if not chunks:
        chunks = [full_resume_text[: settings.llm_max_chunk_chars]]

    logger.info(
        "Long-CV path: %d chunks, total %d chars", len(chunks), len(full_resume_text)
    )

    digest_results = await asyncio.gather(
        *(
            _extract_chunk(chunk, i + 1, len(chunks))
            for i, chunk in enumerate(chunks)
        ),
        return_exceptions=True,
    )
    digests = _coerce_digests(digest_results)
    digest_blob = _format_digest(digests)

    trimmed_jd = _truncate(job_description, settings.llm_max_jd_chars)
    trimmed_digest = _truncate(digest_blob, settings.llm_max_resume_chars)
    user_content = (
        SYSTEM_PROMPT
        + "\n\n"
        + build_user_prompt(trimmed_jd, trimmed_digest)
    )
    logger.info("Final scoring call over aggregated digest (%d chars)", len(trimmed_digest))
    return await _run_screening(user_content, haystack=full_resume_text)


async def _run_screening(
    user_content: str, *, haystack: str
) -> tuple[ScreeningResult, str]:
    settings = get_settings()
    client = get_client()

    try:
        async with _get_semaphore():
            completion = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.0,
                seed=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": SCREENING_JSON_SCHEMA,
                },
            )
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc

    raw = completion.choices[0].message.content or ""
    payload = _parse_json(raw)
    notes = payload.pop("notes", None)
    if notes:
        logger.info("LLM reasoning: %s", notes)

    try:
        result = ScreeningResult.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"LLM response did not match schema: {exc}") from exc

    result = _drop_fabricated_skills(result, haystack)
    result = _align_recommendation(result)
    return result, raw


async def _extract_chunk(chunk_text: str, index: int, total: int) -> ChunkDigest:
    settings = get_settings()
    client = get_client()

    user_content = (
        EXTRACTION_SYSTEM_PROMPT
        + "\n\n"
        + build_extraction_prompt(chunk_text, index, total)
    )

    try:
        async with _get_semaphore():
            completion = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.0,
                seed=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": CHUNK_DIGEST_JSON_SCHEMA,
                },
            )
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        raise LLMError(f"Chunk extraction failed (chunk {index}/{total}): {exc}") from exc

    raw = completion.choices[0].message.content or ""
    payload = _parse_json(raw)
    try:
        return ChunkDigest.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(
            f"Chunk digest did not match schema (chunk {index}/{total}): {exc}"
        ) from exc


def _coerce_digests(results: list) -> list[ChunkDigest]:
    digests: list[ChunkDigest] = []
    for i, outcome in enumerate(results):
        if isinstance(outcome, BaseException):
            logger.warning(
                "Chunk %d extraction failed: %s; proceeding without it",
                i + 1,
                outcome,
            )
            digests.append(ChunkDigest())
        else:
            digests.append(outcome)
    return digests


def _format_digest(digests: list[ChunkDigest]) -> str:
    name = next((d.candidate_name for d in digests if d.candidate_name), "")
    years = max(
        (d.stated_years_experience for d in digests),
        default=0,
    )

    seen_skills: dict[str, str] = {}
    for d in digests:
        for skill in d.skills_found:
            key = skill.strip().lower()
            if key and key not in seen_skills:
                seen_skills[key] = skill.strip()
    skills = list(seen_skills.values())

    seen_roles: set[tuple[str, str, str]] = set()
    roles: list[RoleEntry] = []
    for d in digests:
        for r in d.role_entries:
            key = (
                r.company.strip().lower(),
                r.title.strip().lower(),
                r.start.strip(),
            )
            if key == ("", "", "") or key in seen_roles:
                continue
            seen_roles.add(key)
            roles.append(r)

    lines: list[str] = []
    lines.append(f"Name: {name or '(not detected)'}")
    lines.append(
        "Stated years of experience: "
        + (str(years) if years else "(not stated)")
    )
    lines.append(
        "Skills observed: " + (", ".join(skills) if skills else "(none extracted)")
    )
    lines.append("Roles:")
    if roles:
        for r in roles:
            span = f"{r.start}–{r.end}".strip("–")
            parts = [p for p in (r.title, r.company, span) if p]
            lines.append("  - " + " @ ".join(parts))
    else:
        lines.append("  (none extracted)")
    lines.append("Highlights:")
    for i, d in enumerate(digests):
        highlight = d.highlights.strip() or "(empty)"
        lines.append(f"  --- chunk {i + 1} --- {highlight}")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    logger.info(
        "Truncating input from %d to %d chars for LLM context", len(text), max_chars
    )
    return text[:max_chars] + "\n\n[... truncated ...]"


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise LLMError(f"LLM response contained no JSON object: {raw[:200]!r}")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM response was not valid JSON: {exc}") from exc


def _drop_fabricated_skills(
    result: ScreeningResult, resume_text: str
) -> ScreeningResult:
    haystack = " ".join(resume_text.split()).lower()
    kept: list[str] = []
    dropped: list[str] = []
    for skill in result.matching_skills:
        needle = " ".join(skill.split()).lower()
        if needle and needle in haystack:
            kept.append(skill)
        else:
            dropped.append(skill)

    if not dropped:
        return result

    logger.warning(
        "Dropped fabricated matching_skills not present in resume: %s", dropped
    )
    updated_missing = list(result.missing_skills)
    for skill in dropped:
        if skill not in updated_missing:
            updated_missing.append(skill)
    return result.model_copy(
        update={"matching_skills": kept, "missing_skills": updated_missing}
    )


def _align_recommendation(result: ScreeningResult) -> ScreeningResult:
    expected = "Good fit" if result.match_score >= 70 else "Bad fit"
    if result.recommendation != expected:
        return result.model_copy(update={"recommendation": expected})
    return result


async def check_reachable() -> bool:
    settings = get_settings()
    url = settings.llm_base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                url, headers={"Authorization": f"Bearer {settings.llm_api_key}"}
            )
        return response.status_code < 500
    except httpx.HTTPError:
        return False
