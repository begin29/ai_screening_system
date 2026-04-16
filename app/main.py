from __future__ import annotations

import logging

from fastapi import FastAPI

from app.routers import api, web

logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Resume Screening System",
        version="0.1.0",
        description=(
            "Automated resume screening against a job description using a local "
            "open-source LLM (Mistral) served by LM Studio."
        ),
    )
    app.include_router(api.router)
    app.include_router(web.router)
    return app


app = create_app()
