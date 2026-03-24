"""
Synthetic schema generator for selection-complexity experiments.
Produces question pools with configurable mixed loadings.
"""

import numpy as np
from typing import List


def generate_synthetic_schema(
    n_questions: int = 40,
    n_dims: int = 4,
    mixed_loading_strength: float = 0.3,
    normalize: bool = True,
    random_seed: int = 42,
) -> List[dict]:
    """
    Generate a synthetic question schema with one dominant axis per question
    and optional mixed loadings on other axes.

    Parameters
    ----------
    n_questions : int
        Number of questions.
    n_dims : int
        Number of latent dimensions.
    mixed_loading_strength : float
        Magnitude of off-dominant loadings; sampled from
        Uniform(-mixed_loading_strength, +mixed_loading_strength).
    normalize : bool
        If True, normalize each question's weight vector to unit norm.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    list of dict
        Each dict has 'id' (str) and 'weights' (np.ndarray of length n_dims).
        Compatible with AdaptiveEngine schema["questions"].
    """
    rng = np.random.default_rng(random_seed)
    questions: List[dict] = []

    for i in range(n_questions):
        dominant_axis = i % n_dims
        w = np.zeros(n_dims)
        w[dominant_axis] = 1.0
        for j in range(n_dims):
            if j != dominant_axis:
                w[j] = rng.uniform(-mixed_loading_strength, mixed_loading_strength)
        if normalize:
            nrm = np.linalg.norm(w)
            if nrm > 0:
                w = w / nrm
        questions.append({"id": f"q{i}", "weights": w.astype(float)})

    return questions
