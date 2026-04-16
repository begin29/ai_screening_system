from __future__ import annotations

from app.prompts.screening import SYSTEM_PROMPT, build_user_prompt


def test_build_user_prompt_substitutes_placeholders() -> None:
    prompt = build_user_prompt("Python backend role", "Anu, 5 years")
    assert "Python backend role" in prompt
    assert "Anu, 5 years" in prompt
    assert "BEGIN JOB DESCRIPTION" in prompt
    assert "BEGIN RESUME" in prompt


def test_build_user_prompt_strips_whitespace() -> None:
    prompt = build_user_prompt("   spaced job   \n\n", "\n  spaced resume  \n")
    assert "spaced job" in prompt
    assert "spaced resume" in prompt
    assert "   spaced job" not in prompt
    assert "spaced resume  " not in prompt


def test_build_user_prompt_handles_empty_strings() -> None:
    prompt = build_user_prompt("", "")
    assert "BEGIN JOB DESCRIPTION" in prompt
    assert "BEGIN RESUME" in prompt


def test_system_prompt_describes_scoring_rules() -> None:
    assert "match_score" in SYSTEM_PROMPT
    assert "Good fit" in SYSTEM_PROMPT
    assert "Bad fit" in SYSTEM_PROMPT
