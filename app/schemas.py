from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Recommendation = Literal["Good fit", "Bad fit"]


class ScreeningResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    candidate_name: str = Field(..., min_length=1, max_length=255)
    match_score: int = Field(..., ge=0, le=100)
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience: str = Field(default="")
    recommendation: Recommendation


class ScreeningResponse(BaseModel):
    job_description_id: int
    results: list[ScreeningResult]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_reachable: bool


class RoleEntry(BaseModel):
    title: str = ""
    company: str = ""
    start: str = ""
    end: str = ""


class ChunkDigest(BaseModel):
    candidate_name: str = ""
    stated_years_experience: int = Field(default=0, ge=0)
    skills_found: list[str] = Field(default_factory=list)
    role_entries: list[RoleEntry] = Field(default_factory=list)
    highlights: str = ""
