"""FastAPI backend for adaptive MBTI quiz. Engine state kept server-side; posterior never exposed."""

import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from adaptive_quiz.core import AdaptiveEngine, VarianceSelection, VarianceThresholdStopping
from adaptive_quiz.domains.mbti import mbti_from_posterior

from backend.models import (
    AnswerRequest,
    AnswerResponse,
    QuestionItem,
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
# FRONTEND_DIR env var allows overriding the path (used by Vercel serverless handler).
_frontend_dir = Path(os.environ.get("FRONTEND_DIR", str(Path(__file__).resolve().parent.parent / "frontend")))
if _frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend() -> FileResponse:
        return FileResponse(str(_frontend_dir / "index.html"))


def _engine_uncertainties(engine: Any) -> Dict[str, float]:
    """Per-axis posterior std dev (sqrt of diagonal variances). Axis order: EI, NS, TF, JP."""
    sigma = np.sqrt(np.diag(engine.get_state().Sigma))
    return {"EI": float(sigma[0]), "NS": float(sigma[1]), "TF": float(sigma[2]), "JP": float(sigma[3])}


def _get_question_batch(
    engine: Any,
    question_lookup: Dict[str, Any],
    n: int,
) -> List[QuestionItem]:
    """
    Return up to n next questions ranked by projected variance (w^T Σ w),
    without modifying the engine state.
    """
    state = engine.get_state()
    Sigma = state.Sigma
    excluded_ids = set(state.asked_question_ids)
    batch: List[QuestionItem] = []

    for _ in range(n):
        scores = {}
        for qid, q in question_lookup.items():
            if qid in excluded_ids:
                continue
            w = np.asarray(q["weights"])
            scores[qid] = float(w @ Sigma @ w)
        if not scores:
            break
        best_id = max(scores, key=scores.__getitem__)
        batch.append(QuestionItem(question_id=best_id, prompt=question_lookup[best_id]["text"]))
        excluded_ids.add(best_id)

    return batch


def _session_payload(engine: Any, question_lookup: Dict[str, Any], batch_size: int) -> dict:
    return {
        "engine": engine,
        "question_lookup": question_lookup,
        "batch_size": batch_size,
    }


@app.post("/start_quiz", response_model=StartQuizResponse)
def start_quiz(body: StartQuizRequest) -> StartQuizResponse:
    schema = load_schema(body.schema_name)
    questions = list(schema["questions"])
    random.shuffle(questions)
    schema = {**schema, "questions": questions}
    question_lookup = {q["id"]: q for q in questions}

    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=VarianceSelection(),
        stopping_rule=VarianceThresholdStopping(variance_threshold=0.1),
    )

    session_id = str(uuid.uuid4())
    session_store.create(session_id, _session_payload(engine, question_lookup, body.batch_size))

    next_questions = _get_question_batch(engine, question_lookup, body.batch_size)
    if not next_questions:
        raise HTTPException(status_code=500, detail="No questions available")

    return StartQuizResponse(
        session_id=session_id,
        next_questions=next_questions,
        max_questions=len(schema["questions"]),
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(body: AnswerRequest) -> AnswerResponse:
    session = session_store.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    engine = session["engine"]
    question_lookup = session["question_lookup"]
    batch_size = session.get("batch_size", 5)

    for ans in body.answers:
        if engine.is_complete():
            break
        if ans.question_id not in question_lookup:
            raise HTTPException(status_code=400, detail=f"Unknown question_id: {ans.question_id}")
        engine.submit_answer(ans.question_id, ans.response, ans.response_time)
        session_store.log_response(body.session_id, ans.question_id, ans.response, ans.response_time)

    session_store.update(body.session_id, session)
    questions_asked = engine.get_state().num_questions

    uncertainties = _engine_uncertainties(engine)

    if engine.is_complete():
        state = engine.get_state()
        scored = mbti_from_posterior(state.mu)
        session_store.complete_session(body.session_id, scored.get("type", "UNKNOWN"), state.mu.tolist())
        return AnswerResponse(next_questions=[], complete=True, questions_asked=questions_asked, uncertainties=uncertainties)

    next_questions = _get_question_batch(engine, question_lookup, batch_size)
    complete = len(next_questions) == 0
    if complete:
        state = engine.get_state()
        scored = mbti_from_posterior(state.mu)
        session_store.complete_session(body.session_id, scored.get("type", "UNKNOWN"), state.mu.tolist())

    return AnswerResponse(
        next_questions=next_questions,
        complete=complete,
        questions_asked=questions_asked,
        uncertainties=uncertainties,
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

    dimensions = {
        "E-I": axes.get("EI", 0.0),
        "N-S": axes.get("NS", 0.0),
        "T-F": axes.get("TF", 0.0),
        "J-P": axes.get("JP", 0.0),
    }

    sigma = np.sqrt(np.diag(state["Sigma"]))
    uncertainties = {
        "E-I": float(sigma[0]),
        "N-S": float(sigma[1]),
        "T-F": float(sigma[2]),
        "J-P": float(sigma[3]),
    }

    return ResultResponse(
        type=scored.get("type", "UNKNOWN"),
        dimensions=dimensions,
        uncertainties=uncertainties,
        questions_asked=state["num_questions"],
    )
