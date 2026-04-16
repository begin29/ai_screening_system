"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-15

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "job_descriptions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "resumes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("content_text", sa.Text(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "screenings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "job_description_id",
            sa.Integer(),
            sa.ForeignKey("job_descriptions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "resume_id",
            sa.Integer(),
            sa.ForeignKey("resumes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("candidate_name", sa.String(length=255), nullable=False),
        sa.Column("match_score", sa.Integer(), nullable=False),
        sa.Column("matching_skills", sa.JSON(), nullable=False),
        sa.Column("missing_skills", sa.JSON(), nullable=False),
        sa.Column("experience", sa.String(length=128), nullable=False, server_default=""),
        sa.Column(
            "recommendation",
            sa.Enum("Good fit", "Bad fit", name="recommendation", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("raw_llm_response", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_screenings_job_description_id", "screenings", ["job_description_id"])
    op.create_index("ix_screenings_resume_id", "screenings", ["resume_id"])


def downgrade() -> None:
    op.drop_index("ix_screenings_resume_id", table_name="screenings")
    op.drop_index("ix_screenings_job_description_id", table_name="screenings")
    op.drop_table("screenings")
    op.drop_table("resumes")
    op.drop_table("job_descriptions")
