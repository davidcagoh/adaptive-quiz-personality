"""
Domain-agnostic question selection strategies (variance heuristic, information gain).
No references to MBTI or any questionnaire schema.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Set, Union

import numpy as np


class SelectionStrategy(Protocol):
    """
    Strategy interface for choosing the next question.

    New code should prefer this over the legacy function-based API.
    """

    def select(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        question_weights: Dict[str, np.ndarray],
        asked_ids: Set[str],
    ) -> str:  # returns question_id
        ...


def expected_information_gain(
    Sigma_prior: np.ndarray,
    w: np.ndarray,
    sigma2_t: float,
) -> float:
    """
    Expected information gain (mutual information) for a question.

    Information gain = 0.5 * log(det(Sigma_prior) / det(Sigma_post)),
    where Sigma_post is the posterior covariance after the observation.

    Parameters
    ----------
    Sigma_prior : np.ndarray
        Current posterior covariance (d, d).
    w : np.ndarray
        Question weight vector (d,).
    sigma2_t : float
        Noise variance for the question.

    Returns
    -------
    float
        Expected information gain (nats).
    """
    Sigma_inv_prior = np.linalg.inv(Sigma_prior)
    Sigma_post = np.linalg.inv(Sigma_inv_prior + np.outer(w, w) / sigma2_t)
    det_prior = np.linalg.det(Sigma_prior)
    det_post = np.linalg.det(Sigma_post)
    if det_prior <= 0 or det_post <= 0:
        return 0.0
    return 0.5 * np.log(det_prior / det_post)


def select_next_question(
    mu: np.ndarray,
    Sigma: np.ndarray,
    question_pool: List[np.ndarray],
    asked_indices: Optional[Union[Set[int], List[int]]] = None,
    mode: str = "variance",
    sigma2_t: float = 0.2,
) -> tuple[int, np.ndarray]:
    """
    Select the next question to maximize posterior variance reduction or information gain.

    Parameters
    ----------
    mu : np.ndarray
        Current posterior mean (d,).
    Sigma : np.ndarray
        Current posterior covariance (d, d).
    question_pool : list of np.ndarray
        List of weight vectors, each (d,).
    asked_indices : set or list, optional
        Indices of already asked questions.
    mode : str
        'variance' (projected variance w^T Sigma w) or 'info_gain'.
    sigma2_t : float
        Noise variance used for information gain computation.

    Returns
    -------
    next_index : int
        Index of the selected question in question_pool.
    w_next : np.ndarray
        Weight vector of the selected question.
    """
    if asked_indices is None:
        asked_indices = set()
    asked_set = set(asked_indices)

    scores: List[float] = []
    for idx, w in enumerate(question_pool):
        if idx in asked_set:
            scores.append(-np.inf)
            continue
        if mode == "info_gain":
            score = expected_information_gain(Sigma, w, sigma2_t)
        else:
            score = float(w.T @ Sigma @ w)
        scores.append(score)

    next_index = int(np.argmax(scores))
    w_next = question_pool[next_index]
    return next_index, w_next


class VarianceSelection:
    """Selection strategy that maximizes projected variance w^T Σ w."""

    def select(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        question_weights: Dict[str, np.ndarray],
        asked_ids: Set[str],
    ) -> str:
        scores: Dict[str, float] = {}
        for qid, w in question_weights.items():
            if qid in asked_ids:
                continue
            scores[qid] = float(w.T @ Sigma @ w)
        if not scores:
            raise ValueError("No remaining questions to select from.")
        # argmax over ids – behaviour matches variance heuristic
        best_id = max(scores, key=scores.get)
        return best_id


class InformationGainSelection:
    """Selection strategy that maximizes expected information gain."""

    def __init__(self, sigma2_t: float = 0.2) -> None:
        self.sigma2_t = sigma2_t

    def select(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        question_weights: Dict[str, np.ndarray],
        asked_ids: Set[str],
    ) -> str:
        scores: Dict[str, float] = {}
        for qid, w in question_weights.items():
            if qid in asked_ids:
                continue
            scores[qid] = expected_information_gain(Sigma, w, self.sigma2_t)
        if not scores:
            raise ValueError("No remaining questions to select from.")
        best_id = max(scores, key=scores.get)
        return best_id


def generate_question_weights(
    d: int,
    T: int,
    random_state: Optional[np.random.RandomState] = None,
) -> List[np.ndarray]:
    """
    Generate a list of question weight vectors probing 1 or 2 latent traits.
    Domain-agnostic; used for synthetic pools (e.g. experiments).

    Parameters
    ----------
    d : int
        Number of latent dimensions.
    T : int
        Number of questions.
    random_state : np.random.RandomState, optional
        Random state for reproducibility.

    Returns
    -------
    list of np.ndarray
        List of weight vectors of length T.
    """
    if random_state is None:
        random_state = np.random
    w_list: List[np.ndarray] = []
    for _ in range(T):
        w = np.zeros(d)
        num_axes = random_state.randint(1, 3)
        axes = random_state.choice(d, num_axes, replace=False)
        w[axes] = 1.0
        w_list.append(w)
    return w_list
