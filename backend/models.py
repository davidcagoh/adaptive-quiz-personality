"""Pydantic request/response models for the adaptive quiz API."""

from typing import Dict, Optional

from pydantic import BaseModel


class StartQuizRequest(BaseModel):
    schema_name: str = "mbti"


class StartQuizResponse(BaseModel):
    session_id: str
    question_id: str
    prompt: str
    max_questions: int


class AnswerRequest(BaseModel):
    session_id: str
    response: float


class AnswerResponse(BaseModel):
    question_id: Optional[str] = None
    prompt: Optional[str] = None
    complete: bool
    questions_asked: int = 0


class ResultResponse(BaseModel):
    type: str
    dimensions: Dict[str, float]
    questions_asked: int
