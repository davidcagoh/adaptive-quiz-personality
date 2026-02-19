"""FastAPI backend for adaptive MBTI quiz. Engine state kept server-side; posterior never exposed."""

import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


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

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, etc.
    allow_headers=["*"],
)

AXIS_TO_DIMENSION = {"EI": "E-I", "SN": "S-N", "TF": "T-F", "JP": "J-P"}


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
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(body: AnswerRequest) -> AnswerResponse:
    session = session_store.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    current_question_id = session.get("current_question_id")
    if current_question_id is None:
        # Quiz already complete
        return AnswerResponse(complete=True)

    engine = session["engine"]
    engine.submit_answer(current_question_id, body.response, response_time=1.0)

    if engine.is_complete():
        session["current_question_id"] = None
        return AnswerResponse(complete=True)

    next_q = engine.get_next_question()
    if next_q is None:
        session["current_question_id"] = None
        return AnswerResponse(complete=True)

    question_id, _ = next_q
    session["current_question_id"] = question_id
    return AnswerResponse(
        question_id=question_id,
        prompt=session["question_lookup"][question_id]["text"],
        complete=False,
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
    # Ensure all keys exist
    axes = scored.get("axes", {})
    dimensions = {
        "E-I": axes.get("EI", 0.0),
        "S-N": axes.get("SN", 0.0),
        "T-F": axes.get("TF", 0.0),
        "J-P": axes.get("JP", 0.0),
    }

    resp = ResultResponse(type=scored.get("type", "UNKNOWN"), dimensions=dimensions)
    #session_store.delete(session_id)
    return resp

