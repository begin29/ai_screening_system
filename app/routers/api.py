from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.schemas import HealthResponse, ScreeningResponse
from app.services import llm_client
from app.services.screener import ResumeUpload, screen_resumes
from app.services.text_extractor import (
    ALLOWED_EXTENSIONS,
    ExtractionError,
    FileTooLarge,
    UnsupportedFileType,
)

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    reachable = await llm_client.check_reachable()
    return HealthResponse(status="ok", llm_reachable=reachable)


@router.post("/screen", response_model=ScreeningResponse)
async def screen(
    job_description: str = Form(..., min_length=1),
    resumes: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ScreeningResponse:
    if not resumes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one resume file is required",
        )

    uploads: list[ResumeUpload] = []
    for file in resumes:
        data = await file.read()
        uploads.append(ResumeUpload(filename=file.filename or "resume", data=data))

    try:
        return await screen_resumes(
            job_description=job_description,
            uploads=uploads,
            db=db,
            max_bytes=settings.max_file_bytes,
        )
    except UnsupportedFileType as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"{exc}. Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        ) from exc
    except FileTooLarge as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except ExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
