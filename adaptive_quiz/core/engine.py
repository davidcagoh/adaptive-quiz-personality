"""
Domain-agnostic adaptive engine. Operates in latent vector space only.
No MBTI labels, letter scoring, or hardcoded dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from adaptive_quiz.core.bayesian import bayesian_update
from adaptive_quiz.core.selection import (
    InformationGainSelection,
    SelectionStrategy,
    VarianceSelection,
    select_next_question,
)
from adaptive_quiz.core.stopping import StoppingRule


@dataclass
class EngineState(dict):
    """
    Snapshot of the engine state.

    Inherits from dict for backward compatibility with existing callers that
    use mapping-style access (state["mu"], ...).
    """

    mu: np.ndarray
    Sigma: np.ndarray
    num_questions: int
    asked_question_ids: List[str]

    def __post_init__(self) -> None:
        # Keep mapping-style access in sync with attributes.
        self["mu"] = self.mu
        self["Sigma"] = self.Sigma
        self["num_questions"] = self.num_questions
        self["asked_question_ids"] = self.asked_question_ids


class NoiseModel(Protocol):
    """Protocol for observation noise models."""

    def compute_variance(
        self,
        response: float,
        response_time: Optional[float],
    ) -> float:
        ...


class DefaultLikertNoiseModel:
    """
    Default noise model matching the previous behaviour.

    sigma2_t = 0.1 + 0.1 * response_time + 0.1 * (1 - |response|)
    """

    def compute_variance(
        self,
        response: float,
        response_time: Optional[float],
    ) -> float:
        rt = 0.0 if response_time is None else response_time
        return 0.1 + 0.1 * rt + 0.1 * (1.0 - abs(response))


class _CallableNoiseAdapter(DefaultLikertNoiseModel):
    """
    Adapter to support legacy callable noise_variance_fn while conforming to NoiseModel.
    """

    def __init__(self, fn: Callable[[float, float], float]) -> None:
        self._fn = fn

    def compute_variance(
        self,
        response: float,
        response_time: Optional[float],
    ) -> float:
        rt = 0.0 if response_time is None else response_time
        return self._fn(response, rt)


class AdaptiveEngine:
    """
    Maintains posterior (mu, Sigma), tracks asked questions, selects next question,
    updates posterior, and checks stopping rule. Purely latent space.
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        selection_strategy: Optional[SelectionStrategy | Callable[..., Tuple[int, np.ndarray]]] = None,
        stopping_rule: Optional[StoppingRule] = None,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        selection_mode: str = "variance",
        noise_model: Optional[NoiseModel] = None,
        noise_variance_fn: Optional[Callable[[float, float], float]] = None,
        on_update: Optional[Callable[[EngineState], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        schema : dict
            Must have "questions" (list of {"id", "weights", ...}) and optionally
            "dimensions" for dimension count. Weights define the question pool.
        selection_strategy : SelectionStrategy or callable, optional
            New style: object implementing SelectionStrategy.select(mu, Sigma, question_weights, asked_ids) -> question_id.
            Legacy: callable (mu, Sigma, question_pool, asked_indices, **kwargs) -> (index, weight_vector).
            If None, uses built-in variance/info_gain strategy based on selection_mode.
        stopping_rule : StoppingRule, optional
            If None, engine never stops by rule (caller can use max_questions via get_state).
        prior_mean : np.ndarray, optional
            Prior mean (d,). Default: zeros.
        prior_cov : np.ndarray, optional
            Prior covariance (d, d). Default: identity.
        selection_mode : str
            Used when selection_strategy is None: 'variance' or 'info_gain'.
        noise_model : NoiseModel, optional
            Observation noise model. If None, uses DefaultLikertNoiseModel unless
            noise_variance_fn is provided (legacy path).
        noise_variance_fn : callable, optional
            Legacy hook (response, response_time) -> sigma2_t. If provided and
            noise_model is None, it is wrapped in a NoiseModel adapter.
        on_update : callable, optional
            Optional hook called with EngineState after each successful submit_answer.
        """
        questions = schema["questions"]
        self._questions = questions
        self._weight_vectors = [np.asarray(q["weights"], dtype=float) for q in questions]
        self._id_to_index = {q["id"]: i for i, q in enumerate(questions)}
        self._id_to_weight: Dict[str, np.ndarray] = {
            q["id"]: self._weight_vectors[i] for i, q in enumerate(questions)
        }
        self._d = len(self._weight_vectors[0])
        self._selection_strategy = selection_strategy
        self._stopping_rule = stopping_rule
        self._selection_mode = selection_mode
        if noise_model is not None:
            self._noise_model: NoiseModel = noise_model
        elif noise_variance_fn is not None:
            self._noise_model = _CallableNoiseAdapter(noise_variance_fn)
        else:
            self._noise_model = DefaultLikertNoiseModel()
        self._on_update = on_update

        self._mu = prior_mean if prior_mean is not None else np.zeros(self._d)
        self._Sigma = prior_cov if prior_cov is not None else np.eye(self._d)
        self._asked_indices: set = set()
        self._max_questions = len(questions)

    def get_next_question(self) -> Optional[Tuple[str, np.ndarray]]:
        """
        Returns the next question to ask, or None if complete or no questions left.

        Returns
        -------
        (question_id, weight_vector) or None
        """
        if self.is_complete():
            return None
        if self._selection_strategy is not None:
            # Support both new SelectionStrategy objects and legacy callables.
            if hasattr(self._selection_strategy, "select"):
                asked_ids = {self._questions[i]["id"] for i in self._asked_indices}
                question_id = self._selection_strategy.select(
                    self._mu,
                    self._Sigma,
                    self._id_to_weight,
                    asked_ids,
                )
                idx = self._id_to_index[question_id]
                w = self._weight_vectors[idx]
            else:
                # Legacy function-based API.
                idx, w = self._selection_strategy(
                    self._mu,
                    self._Sigma,
                    self._weight_vectors,
                    self._asked_indices,
                )
                question_id = self._questions[idx]["id"]
        else:
            # Built-in strategies: keep behaviour equivalent to original.
            if self._selection_mode == "info_gain":
                strategy = InformationGainSelection()
                asked_ids = {self._questions[i]["id"] for i in self._asked_indices}
                question_id = strategy.select(
                    self._mu,
                    self._Sigma,
                    self._id_to_weight,
                    asked_ids,
                )
                idx = self._id_to_index[question_id]
                w = self._weight_vectors[idx]
            else:
                # Default variance-based heuristics via legacy helper for exact behaviour.
                idx, w = select_next_question(
                    self._mu,
                    self._Sigma,
                    self._weight_vectors,
                    self._asked_indices,
                    mode="variance",
                )
                question_id = self._questions[idx]["id"]
        return question_id, w.copy()

    def submit_answer(
        self,
        question_id: str,
        response: float,
        response_time: float,
    ) -> None:
        """
        Record answer and update posterior.

        Parameters
        ----------
        question_id : str
            Id of the question that was answered.
        response : float
            Observed response (e.g. Likert value).
        response_time : float
            Response time (used for noise model).
        """
        if question_id not in self._id_to_index:
            raise ValueError(f"Unknown question_id: {question_id}")
        idx = self._id_to_index[question_id]
        w = self._weight_vectors[idx]
        sigma2_t = self._noise_model.compute_variance(response, response_time)
        self._mu, self._Sigma = bayesian_update(
            self._mu, self._Sigma, w, response, sigma2_t
        )
        self._asked_indices.add(idx)
        if self._on_update is not None:
            self._on_update(self.get_state())

    def is_complete(self) -> bool:
        """True if stopping rule says stop or all questions asked."""
        if self._stopping_rule is None:
            return len(self._asked_indices) >= self._max_questions
        return self._stopping_rule.should_stop(
            self._mu,
            self._Sigma,
            num_asked=len(self._asked_indices),
            max_questions=self._max_questions,
        )

    def get_state(self) -> EngineState:
        """
        Current engine state for inspection or persistence.

        Returns
        -------
        EngineState
            Snapshot with mu, Sigma, num_questions, asked_question_ids.
        """
        asked_ids = [self._questions[i]["id"] for i in sorted(self._asked_indices)]
        return EngineState(
            mu=self._mu.copy(),
            Sigma=self._Sigma.copy(),
            num_questions=len(self._asked_indices),
            asked_question_ids=asked_ids,
        )
