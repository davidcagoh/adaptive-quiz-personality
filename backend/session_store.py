"""
Session store with two implementations:
  - MemorySessionStore: in-memory dict, default for local dev.
  - SupabaseSessionStore: persists across restarts and serverless instances.

Selected automatically based on SUPABASE_URL / SUPABASE_KEY env vars.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

_SUPABASE_URL = os.environ.get("SUPABASE_URL")
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_payload(payload: Dict[str, Any]) -> Dict:
    """Convert a session payload (containing an AdaptiveEngine) to a JSON-safe dict."""
    engine = payload["engine"]
    state = engine.get_state()
    return {
        "mu": state.mu.tolist(),
        "Sigma": state.Sigma.tolist(),
        "asked_question_ids": state.asked_question_ids,
        "current_question_id": payload.get("current_question_id"),
        "schema_name": payload.get("schema_name", "mbti"),
    }


def _deserialize_payload(data: Dict) -> Dict[str, Any]:
    """Reconstruct a session payload (with a live AdaptiveEngine) from a serialized dict."""
    from adaptive_quiz.core import AdaptiveEngine, VarianceSelection, VarianceThresholdStopping

    # Lazy import to avoid circular dependency at module load time.
    from backend.schemas import load_schema

    schema_name = data.get("schema_name", "mbti")
    schema = load_schema(schema_name)
    question_lookup = {q["id"]: q for q in schema["questions"]}

    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=VarianceSelection(),
        stopping_rule=VarianceThresholdStopping(variance_threshold=0.1),
        prior_mean=np.array(data["mu"]),
        prior_cov=np.array(data["Sigma"]),
    )
    # Restore which questions have been asked by patching internal state.
    engine._asked_indices = {engine._id_to_index[qid] for qid in data["asked_question_ids"]}

    return {
        "engine": engine,
        "current_question_id": data.get("current_question_id"),
        "question_lookup": question_lookup,
        "schema_name": schema_name,
    }


# ---------------------------------------------------------------------------
# In-memory store (local dev)
# ---------------------------------------------------------------------------

class MemorySessionStore:
    """Sessions held in process memory. Lost on restart."""

    def __init__(self) -> None:
        self._sessions: dict = {}

    def create(self, session_id: str, payload: Dict[str, Any]) -> None:
        self._sessions[session_id] = payload

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def update(self, session_id: str, payload: Dict[str, Any]) -> None:
        self._sessions[session_id] = payload

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def log_response(self, session_id: str, question_id: str, response: float, response_time: float) -> None:
        pass  # no-op for local dev

    def complete_session(self, session_id: str, final_type: str, final_mu: List[float]) -> None:
        pass  # no-op for local dev


# ---------------------------------------------------------------------------
# Supabase-backed store (production)
# ---------------------------------------------------------------------------

class SupabaseSessionStore:
    """
    Sessions stored in Supabase. Safe for serverless — no in-process state.

    Required env vars:
        SUPABASE_URL  — e.g. https://<ref>.supabase.co
        SUPABASE_KEY  — service role key (preferred) or anon key
    """

    def __init__(self) -> None:
        from supabase import create_client
        self._db = create_client(_SUPABASE_URL, _SUPABASE_KEY)

    def create(self, session_id: str, payload: Dict[str, Any]) -> None:
        self._db.table("sessions").insert({
            "id": session_id,
            "engine_state": _serialize_payload(payload),
        }).execute()

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        result = (
            self._db.table("sessions")
            .select("engine_state")
            .eq("id", session_id)
            .execute()
        )
        if not result.data:
            return None
        return _deserialize_payload(result.data[0]["engine_state"])

    def update(self, session_id: str, payload: Dict[str, Any]) -> None:
        self._db.table("sessions").update({
            "engine_state": _serialize_payload(payload),
        }).eq("id", session_id).execute()

    def delete(self, session_id: str) -> None:
        self._db.table("sessions").delete().eq("id", session_id).execute()

    def log_response(self, session_id: str, question_id: str, response: float, response_time: float) -> None:
        self._db.table("responses").insert({
            "session_id": session_id,
            "question_id": question_id,
            "response": response,
            "response_time": response_time,
        }).execute()

    def complete_session(self, session_id: str, final_type: str, final_mu: List[float]) -> None:
        self._db.table("sessions").update({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "final_type": final_type,
            "final_mu": final_mu,
        }).eq("id", session_id).execute()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _make_store():
    if _SUPABASE_URL and _SUPABASE_KEY:
        return SupabaseSessionStore()
    return MemorySessionStore()


session_store = _make_store()
