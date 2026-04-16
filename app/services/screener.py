from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models import JobDescription, Recommendation, Resume, Screening
from app.schemas import ScreeningResponse, ScreeningResult
from app.services import llm_client
from app.services.text_extractor import extract_pages

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeUpload:
    filename: str
    data: bytes


async def screen_resumes(
    job_description: str,
    uploads: list[ResumeUpload],
    db: Session,
    max_bytes: int | None = None,
) -> ScreeningResponse:
    if not uploads:
        raise ValueError("At least one resume is required")

    jd_row = JobDescription(content=job_description)
    db.add(jd_row)
    db.flush()

    extracted: list[tuple[int, str, str, list[str]]] = []
    for upload in uploads:
        pages = extract_pages(upload.filename, upload.data, max_bytes=max_bytes)
        text = "\n\n".join(pages)
        resume_row = Resume(filename=upload.filename, content_text=text)
        db.add(resume_row)
        db.flush()
        extracted.append((resume_row.id, upload.filename, text, pages))

    jd_id = jd_row.id
    db.commit()

    results = await asyncio.gather(
        *(
            llm_client.screen_one(job_description, text, pages=pages)
            for _, _, text, pages in extracted
        ),
        return_exceptions=True,
    )

    screening_results: list[ScreeningResult] = []
    for (resume_id, filename, _, _), outcome in zip(extracted, results):
        if isinstance(outcome, BaseException):
            logger.warning("LLM screening failed for %s: %s", filename, outcome)
            result = _fallback_result(filename)
            raw_response = str(outcome)
        else:
            result, raw_response = outcome

        db.add(
            Screening(
                job_description_id=jd_id,
                resume_id=resume_id,
                candidate_name=result.candidate_name,
                match_score=result.match_score,
                matching_skills=list(result.matching_skills),
                missing_skills=list(result.missing_skills),
                experience=result.experience,
                recommendation=Recommendation(result.recommendation),
                raw_llm_response=raw_response,
            )
        )
        screening_results.append(result)

    db.commit()

    screening_results.sort(key=lambda r: r.match_score, reverse=True)
    return ScreeningResponse(job_description_id=jd_id, results=screening_results)


def _fallback_result(filename: str) -> ScreeningResult:
    return ScreeningResult(
        candidate_name=filename,
        match_score=0,
        matching_skills=[],
        missing_skills=[],
        experience="Unknown",
        recommendation="Bad fit",
    )
