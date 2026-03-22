"""FastAPI backend for adaptive MBTI quiz. Engine state kept server-side; posterior never exposed."""

import os
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


from adaptive_quiz.core import AdaptiveEngine, SignConfidenceStopping, VarianceSelection
from adaptive_quiz.domains.mbti import mbti_from_posterior

from backend.models import (
    AnswerRequest,
    AnswerResponse,
    ResultResponse,
    StartQuizRequest,
    StartQuizResponse,
)
from backend.schemas import load_schema
from backend.session_store import session_store

app = FastAPI()

# CORS: allow comma-separated origins from env or fall back to localhost:3000.
_cors_origins_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend if the directory exists (optional — quiz works without it).
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend() -> FileResponse:
        return FileResponse(str(_frontend_dir / "index.html"))


def _session_payload(engine: Any, current_question_id: str, question_lookup: Dict[str, Any]) -> dict:
    return {
        "engine": engine,
        "current_question_id": current_question_id,
        "question_lookup": question_lookup,
    }


@app.post("/start_quiz", response_model=StartQuizResponse)
def start_quiz(body: StartQuizRequest) -> StartQuizResponse:
    schema = load_schema(body.schema_name)
    question_lookup = {q["id"]: q for q in schema["questions"]}

    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=VarianceSelection(),
        stopping_rule=SignConfidenceStopping(confidence_threshold=0.95),
    )
    next_q = engine.get_next_question()
    if next_q is None:
        raise HTTPException(status_code=500, detail="No question available")

    question_id, _ = next_q
    session_id = str(uuid.uuid4())
    session_store.create(session_id, _session_payload(engine, question_id, question_lookup))

    return StartQuizResponse(
        session_id=session_id,
        question_id=question_id,
        prompt=question_lookup[question_id]["text"],
        max_questions=len(schema["questions"]),
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(body: AnswerRequest) -> AnswerResponse:
    session = session_store.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    current_question_id = session.get("current_question_id")
    if current_question_id is None:
        # Quiz already complete — report current count.
        engine = session["engine"]
        return AnswerResponse(complete=True, questions_asked=engine.get_state()["num_questions"])

    engine = session["engine"]
    engine.submit_answer(current_question_id, body.response, response_time=1.0)
    questions_asked = engine.get_state()["num_questions"]

    if engine.is_complete():
        session["current_question_id"] = None
        return AnswerResponse(complete=True, questions_asked=questions_asked)

    next_q = engine.get_next_question()
    if next_q is None:
        session["current_question_id"] = None
        return AnswerResponse(complete=True, questions_asked=questions_asked)

    question_id, _ = next_q
    session["current_question_id"] = question_id
    return AnswerResponse(
        question_id=question_id,
        prompt=session["question_lookup"][question_id]["text"],
        complete=False,
        questions_asked=questions_asked,
    )


@app.get("/result", response_model=ResultResponse)
def result(session_id: str) -> ResultResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    engine = session["engine"]
    if not engine.is_complete():
        raise HTTPException(status_code=400, detail="Quiz not complete")

    state = engine.get_state()
    scored = mbti_from_posterior(state.mu)
    axes = scored.get("axes", {})

    # Convention: first letter of each key = positive-mu letter.
    # E-I: positive → E  |  N-S: positive → N  |  T-F: positive → T  |  J-P: positive → J
    dimensions = {
        "E-I": axes.get("EI", 0.0),
        "N-S": axes.get("NS", 0.0),
        "T-F": axes.get("TF", 0.0),
        "J-P": axes.get("JP", 0.0),
    }

    # Sessions are in-memory only; we intentionally keep the session alive so
    # /result can be called multiple times (e.g. on page refresh) without a 404.
    # Sessions are automatically cleared on server restart.
    return ResultResponse(
        type=scored.get("type", "UNKNOWN"),
        dimensions=dimensions,
        questions_asked=state["num_questions"],
    )
