"""
MBTI scoring: convert posterior to MBTI type, axis scores, and confidence.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def mbti_from_posterior(mu: np.ndarray) -> Dict[str, Any]:
    """
    Convert posterior mean vector to MBTI type and raw axis scores.

    Parameters
    ----------
    mu : np.ndarray
        Posterior mean vector of length 4: [EI, NS, TF, JP]

    Returns
    -------
    dict
        {
            "axes": {"EI": float, "NS": float, "TF": float, "JP": float},
            "type": str  # e.g. "INTJ"
        }
    """
    if len(mu) != 4:
        raise ValueError(f"Expected mu of length 4, got {len(mu)}")

    ei_val, sn_val, tf_val, jp_val = float(mu[0]), float(mu[1]), float(mu[2]), float(mu[3])

    # Positive mu -> positive (first) letter per schema convention
    mbti_type = (
        ("E" if ei_val >= 0 else "I")
        + ("N" if sn_val >= 0 else "S")
        + ("T" if tf_val >= 0 else "F")
        + ("J" if jp_val >= 0 else "P")
    )

    return {
        "axes": {"EI": ei_val, "NS": sn_val, "TF": tf_val, "JP": jp_val},
        "type": mbti_type,
    }


class MBTIScorer:
    """
    Maps posterior mean (and optionally covariance) to MBTI type, axis scores, and confidence.
    Reads dimension metadata from the schema (positive/negative letter per axis).
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters
        ----------
        schema : dict, optional
            MBTI schema with "dimensions" list. If None, loads default mbti.json.
        """
        if schema is None:
            from adaptive_quiz.domains.mbti.schema import load_mbti_schema
            schema = load_mbti_schema()
        self._dimensions = schema["dimensions"]

    def score(
        self,
        mu: np.ndarray,
        Sigma: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Map latent mean (and optional covariance) to MBTI result.

        Parameters
        ----------
        mu : np.ndarray
            Posterior mean (d,).
        Sigma : np.ndarray, optional
            Posterior covariance (d, d). If provided, confidence uses sqrt(diag(Sigma)).

        Returns
        -------
        dict
            - type: str, e.g. "INTJ"
            - letters: list of str, one per axis
            - axis_scores: list of float (preference strength 0–1)
            - confidence: list of float (per-axis posterior std; lower = more confident)
        """
        d = len(mu)
        letters: List[str] = []
        axis_scores: List[float] = []
        confidence: List[float] = []

        for i in range(min(d, len(self._dimensions))):
            dim = self._dimensions[i]
            letters.append(dim["positive"] if mu[i] >= 0 else dim["negative"])
            axis_scores.append(float(min(1.0, abs(mu[i]))))
            if Sigma is not None and i < Sigma.shape[0]:
                confidence.append(float(np.sqrt(Sigma[i, i])))
            else:
                confidence.append(0.0)

        return {
            "type": "".join(letters),
            "letters": letters,
            "axis_scores": axis_scores,
            "confidence": confidence,
        }
