from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import JobDescription, JobTemplate, Screening
from app.schemas import ScreeningResponse, ScreeningResult
from app.services.screener import ResumeUpload, screen_resumes
from app.services.text_extractor import (
    ALLOWED_EXTENSIONS,
    ExtractionError,
    FileTooLarge,
    UnsupportedFileType,
)

router = APIRouter(tags=["web"])

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

RECENT_SCANS_LIMIT = 10


def _fetch_recent_scans(db: Session, limit: int = RECENT_SCANS_LIMIT) -> list[dict]:
    stmt = (
        select(
            JobDescription.id,
            JobDescription.content,
            JobDescription.created_at,
            func.count(Screening.id).label("resume_count"),
            func.max(Screening.match_score).label("top_score"),
        )
        .outerjoin(Screening, Screening.job_description_id == JobDescription.id)
        .group_by(JobDescription.id)
        .order_by(JobDescription.created_at.desc())
        .limit(limit)
    )
    rows = db.execute(stmt).all()
    return [
        {
            "id": row.id,
            "preview": _preview(row.content),
            "created_at": row.created_at,
            "resume_count": row.resume_count or 0,
            "top_score": row.top_score,
        }
        for row in rows
    ]


def _preview(text: str, length: int = 110) -> str:
    clean = " ".join(text.split())
    return clean[:length] + ("..." if len(clean) > length else "")


def _fetch_templates(db: Session) -> list[JobTemplate]:
    return list(
        db.execute(select(JobTemplate).order_by(JobTemplate.name)).scalars()
    )


def _render_index(
    request: Request,
    db: Session,
    *,
    error: str | None = None,
    template_error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    job_templates = _fetch_templates(db)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "accepted": sorted(ALLOWED_EXTENSIONS),
            "recent_scans": _fetch_recent_scans(db),
            "job_templates": job_templates,
            "templates_payload": [
                {"id": t.id, "name": t.name, "content": t.content}
                for t in job_templates
            ],
            "error": error,
            "template_error": template_error,
        },
        status_code=status_code,
    )


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render_index(request, db)


@router.post("/screen", response_class=HTMLResponse)
async def screen_form(
    request: Request,
    job_description: str = Form(..., min_length=1),
    resumes: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    uploads = [
        ResumeUpload(filename=f.filename or "resume", data=await f.read())
        for f in resumes
    ]

    try:
        response = await screen_resumes(
            job_description=job_description,
            uploads=uploads,
            db=db,
            max_bytes=settings.max_file_bytes,
        )
    except (UnsupportedFileType, FileTooLarge, ExtractionError) as exc:
        return _render_index(request, db, error=str(exc), status_code=400)

    return templates.TemplateResponse(
        request,
        "results.html",
        {"response": response, "job_description": job_description},
    )


@router.post("/templates", response_class=HTMLResponse)
async def create_template(
    request: Request,
    name: str = Form(..., min_length=1, max_length=255),
    content: str = Form(..., min_length=1),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    template = JobTemplate(name=name.strip(), content=content.strip())
    db.add(template)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return _render_index(
            request,
            db,
            template_error=f"A template named {name!r} already exists.",
            status_code=400,
        )
    return RedirectResponse(url="/", status_code=303)


@router.post("/templates/{template_id}/delete")
async def delete_template(
    template_id: int,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    template = db.get(JobTemplate, template_id)
    if template is not None:
        db.delete(template)
        db.commit()
    return RedirectResponse(url="/", status_code=303)


@router.get("/scans/{job_description_id}", response_class=HTMLResponse)
async def view_scan(
    request: Request,
    job_description_id: int,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    jd = db.get(JobDescription, job_description_id)
    if jd is None:
        raise HTTPException(status_code=404, detail="Scan not found")

    stmt = (
        select(Screening)
        .where(Screening.job_description_id == job_description_id)
        .order_by(Screening.match_score.desc())
    )
    screenings = db.execute(stmt).scalars().all()

    results = [
        ScreeningResult(
            candidate_name=s.candidate_name,
            match_score=s.match_score,
            matching_skills=s.matching_skills,
            missing_skills=s.missing_skills,
            experience=s.experience,
            recommendation=s.recommendation.value,
        )
        for s in screenings
    ]
    response = ScreeningResponse(job_description_id=jd.id, results=results)

    return templates.TemplateResponse(
        request,
        "results.html",
        {"response": response, "job_description": jd.content},
    )
