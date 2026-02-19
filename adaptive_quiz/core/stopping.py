"""
Domain-agnostic stopping rules for the adaptive engine.
No references to MBTI or any questionnaire schema.
"""

import math
from typing import Optional, Protocol

import numpy as np


class StoppingRule(Protocol):
    """Protocol for stopping rules."""

    def should_stop(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        num_asked: int,
        max_questions: int,
    ) -> bool:
        """Return True if the assessment should stop."""
        ...


class VarianceThresholdStopping:
    """
    Stop when max diagonal of Sigma is below threshold, or max_questions reached.
    """

    def __init__(
        self,
        variance_threshold: float = 0.1,
        max_questions: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        variance_threshold : float
            Stop when max(diag(Sigma)) < this value.
        max_questions : int, optional
            Hard cap on number of questions; if None, only variance threshold applies.
        """
        self.variance_threshold = variance_threshold
        self.max_questions = max_questions

    def should_stop(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        num_asked: int,
        max_questions: int,
    ) -> bool:
        """
        Parameters
        ----------
        mu : np.ndarray
            Current posterior mean (d,). Not used by this rule.
        Sigma : np.ndarray
            Current posterior covariance (d, d).
        num_asked : int
            Number of questions asked so far.
        max_questions : int
            Maximum questions allowed (e.g. from schema or config).

        Returns
        -------
        bool
            True if assessment should stop.
        """
        if self.max_questions is not None and num_asked >= self.max_questions:
            return True
        if max_questions is not None and num_asked >= max_questions:
            return True
        max_var = float(np.max(np.diag(Sigma)))
        return max_var < self.variance_threshold


class SignConfidenceStopping:
    """
    Stop when posterior sign confidence exceeds threshold for all dimensions.

    For each dimension i:
        - Compute z_i = mu_i / std_i where std_i = sqrt(Sigma_ii)
        - Compute p_pos = Φ(z_i) and p_neg = 1 - p_pos
        - confidence_i = max(p_pos, p_neg)
    Stop if confidence_i >= threshold for ALL dimensions.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        max_questions: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        confidence_threshold : float
            Stop when sign confidence >= this value for all dimensions.
        max_questions : int, optional
            Hard cap on number of questions; if None, only confidence threshold applies.
        """
        self.confidence_threshold = confidence_threshold
        self.max_questions = max_questions

    def _normal_cdf(self, z: float) -> float:
        """Compute standard normal CDF: Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def should_stop(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        num_asked: int,
        max_questions: int,
    ) -> bool:
        """
        Parameters
        ----------
        mu : np.ndarray
            Current posterior mean (d,).
        Sigma : np.ndarray
            Current posterior covariance (d, d).
        num_asked : int
            Number of questions asked so far.
        max_questions : int
            Maximum questions allowed (e.g. from schema or config).

        Returns
        -------
        bool
            True if assessment should stop.
        """
        if self.max_questions is not None and num_asked >= self.max_questions:
            return True
        if max_questions is not None and num_asked >= max_questions:
            return True

        d = len(mu)
        for i in range(d):
            var_i = Sigma[i, i]
            if var_i <= 0.0:
                # Edge case: zero or negative variance means perfect confidence
                continue

            std_i = math.sqrt(var_i)
            z_i = mu[i] / std_i
            p_pos = self._normal_cdf(z_i)
            p_neg = 1.0 - p_pos
            confidence = max(p_pos, p_neg)

            if confidence < self.confidence_threshold:
                return False

        return True
