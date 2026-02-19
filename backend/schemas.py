"""Pluggable schema loader. Future schemas register here."""

from fastapi import HTTPException

from adaptive_quiz.domains.mbti import load_mbti_schema


def load_schema(schema_name: str) -> dict:
    if schema_name == "mbti":
        return load_mbti_schema()
    raise HTTPException(status_code=404, detail=f"Unknown schema: {schema_name}")
