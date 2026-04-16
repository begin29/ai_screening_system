from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_screening.db")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:1234/v1")
os.environ.setdefault("LLM_MODEL", "mistral-test")


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import database, models  # noqa: F401

    db_file = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    test_sessionmaker = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )

    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", test_sessionmaker)

    database.Base.metadata.create_all(bind=engine)
    try:
        yield
    finally:
        database.Base.metadata.drop_all(bind=engine)
        engine.dispose()
