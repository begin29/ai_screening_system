# AI Resume Screening System

Automated first-pass resume screening against a job description using a local
open-source LLM (Mistral) served by [LM Studio](https://lmstudio.ai/).

Given a job description and a batch of resumes, the system extracts the resume
text, sends each pair to the local model, and returns a ranked list with
match score, matching/missing skills, experience, and a **Good fit** /
**Bad fit** recommendation.

## Stack

- Python 3.12+
- FastAPI + Uvicorn
- `openai` Python SDK pointed at LM Studio's OpenAI-compatible endpoint
- SQLAlchemy 2.x + Alembic (SQLite now, Postgres-ready)
- `pdfplumber`, `python-docx`, LibreOffice (for `.doc`)
- Jinja2 templates + Tailwind (CDN, no build step)

## Prerequisites

- **Python 3.12+**
- **LM Studio** running locally with a Mistral model loaded and the local
  server started (default `http://localhost:1234/v1`)
- **LibreOffice** installed **only if you need `.doc` parsing**
  (`brew install --cask libreoffice` on macOS). PDF, DOCX, and TXT work from
  pip packages alone.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
```

Edit `.env` if LM Studio is on a non-default host/port or a different model is
loaded.

## Run

```bash
uvicorn app.main:app --reload
```

Then:

- `http://localhost:8000/` — upload form (paste JD, upload resumes, submit).
- `http://localhost:8000/docs` — Swagger UI for the REST API.
- `http://localhost:8000/api/health` — includes `llm_reachable` so you
  can sanity-check connectivity to your LLM backend.

## REST API

### `POST /api/screen`

Multipart form:

- `job_description` — text (required)
- `resumes` — one or more files (`.pdf`, `.doc`, `.docx`, `.txt`)

Example:

```bash
curl -X POST http://localhost:8000/api/screen \
    -F "job_description=Senior Python backend engineer with FastAPI" \
    -F "resumes=@./anu.pdf" \
    -F "resumes=@./bob.docx"
```

Response (matches the project spec):

```json
{
  "job_description_id": 1,
  "results": [
    {
      "candidate_name": "Anu",
      "match_score": 85,
      "matching_skills": ["Python", "FastAPI"],
      "missing_skills": ["Docker"],
      "experience": "5",
      "recommendation": "Good fit"
    }
  ]
}
```

Results are sorted by `match_score` descending.

## Tests

```bash
pytest
```

The test suite stubs the LM Studio call, so tests run offline.

## Future: Postgres migration

The application code is written to be dialect-agnostic. When moving to
Postgres, all that's needed is:

1. `pip install "psycopg[binary]"`
2. Set `DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname` in `.env`
3. `alembic upgrade head` against the new database
4. One-off data migration from `screening.db` if existing rows need to be
   carried over (not a code change)

The rules that keep this promise intact are listed in the plan file and boil
down to: only use generic SQLAlchemy types, no raw SQL, `DateTime(timezone=True)`
everywhere, enums as `native_enum=False`, and all schema changes through
Alembic.
