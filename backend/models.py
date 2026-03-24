"""Pydantic request/response models for the adaptive quiz API."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class StartQuizRequest(BaseModel):
    schema_name: str = "mbti"
    batch_size: int = 5


class QuestionItem(BaseModel):
    question_id: str
    prompt: str


class StartQuizResponse(BaseModel):
    session_id: str
    next_questions: List[QuestionItem]
    max_questions: int


class SingleAnswer(BaseModel):
    question_id: str
    response: float
    response_time: float = 1.0


class AnswerRequest(BaseModel):
    session_id: str
    answers: List[SingleAnswer]


class AnswerResponse(BaseModel):
    next_questions: List[QuestionItem]
    complete: bool
    questions_asked: int = 0
    uncertainties: Dict[str, float] = {}  # per-axis posterior std dev (EI, NS, TF, JP)


class ResultResponse(BaseModel):
    type: str
    dimensions: Dict[str, float]
    uncertainties: Dict[str, float]  # per-axis posterior std dev
    questions_asked: int
