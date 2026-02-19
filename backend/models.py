"""Pydantic request/response models for the adaptive quiz API."""

from typing import Dict, Optional

from pydantic import BaseModel


class StartQuizRequest(BaseModel):
    schema_name: str = "mbti"


class StartQuizResponse(BaseModel):
    session_id: str
    question_id: str
    prompt: str


class AnswerRequest(BaseModel):
    session_id: str
    response: float


class AnswerResponse(BaseModel):
    question_id: Optional[str] = None
    prompt: Optional[str] = None
    complete: bool


class ResultResponse(BaseModel):
    type: str
    dimensions: Dict[str, float]
