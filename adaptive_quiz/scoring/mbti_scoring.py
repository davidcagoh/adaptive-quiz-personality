"""
MBTI-specific scoring: map latent mu to type string, axis scores, and confidence.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def load_mbti_schema(schema_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load MBTI schema from JSON; if path is None, use default schemas/mbti.json."""
    if schema_path is None:
        schema_path = Path(__file__).parent.parent / "schemas" / "mbti.json"
    with open(schema_path) as f:
        return json.load(f)


class MBTIScorer:
    """
    Maps posterior mean (and optionally covariance) to MBTI type, axis scores, and confidence.
    Engine does not call this; scoring is separate.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters
        ----------
        schema : dict, optional
            MBTI schema with "dimensions" (name, positive, negative). If None, loads default mbti.json.
        """
        self._schema = schema if schema is not None else load_mbti_schema()
        self._dimensions = self._schema["dimensions"]

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
            Posterior mean (d,) in latent space.
        Sigma : np.ndarray, optional
            Posterior covariance (d, d). If provided, confidence uses sqrt(diag(Sigma)).

        Returns
        -------
        dict
            - type: str, e.g. "INTJ"
            - axis_scores: list of float (strength of preference per axis, 0–1)
            - confidence: list of float (per-axis, lower = more confident) if Sigma given
            - letters: list of str, one per axis (e.g. ["I", "N", "T", "J"])
        """
        d = len(mu)
        letters: List[str] = []
        axis_scores: List[float] = []
        confidence: List[float] = []

        for i in range(min(d, len(self._dimensions))):
            dim = self._dimensions[i]
            positive = dim["positive"]
            negative = dim["negative"]
            # sign(mu_i) -> letter: positive mu -> positive letter
            if mu[i] >= 0:
                letters.append(positive)
            else:
                letters.append(negative)
            # Strength of preference from magnitude (clip to [0,1] for scale)
            strength = min(1.0, abs(mu[i]))
            axis_scores.append(float(strength))
            if Sigma is not None and i < Sigma.shape[0]:
                # Confidence: lower std = more confident
                std = np.sqrt(Sigma[i, i])
                confidence.append(float(std))
            else:
                confidence.append(0.0)

        type_str = "".join(letters)
        return {
            "type": type_str,
            "letters": letters,
            "axis_scores": axis_scores,
            "confidence": confidence,
        }
