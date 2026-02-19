"""
MBTI scoring: convert posterior mean vector to MBTI type and axis scores.
"""

from typing import Dict

import numpy as np


def mbti_from_posterior(mu: np.ndarray) -> Dict[str, any]:
    """
    Convert posterior mean vector to MBTI type and axis scores.

    Parameters
    ----------
    mu : np.ndarray
        Posterior mean vector of length 4: [EI, SN, TF, JP]

    Returns
    -------
    dict
        {
            "axes": {
                "EI": float,
                "SN": float,
                "TF": float,
                "JP": float
            },
            "type": str  # e.g., "INTJ"
        }
    """
    if len(mu) != 4:
        raise ValueError(f"Expected mu of length 4, got {len(mu)}")

    ei_val = float(mu[0])
    sn_val = float(mu[1])
    tf_val = float(mu[2])
    jp_val = float(mu[3])

    # Map each axis: >= 0 -> first letter, < 0 -> second letter
    ei_letter = "E" if ei_val >= 0 else "I"
    sn_letter = "S" if sn_val >= 0 else "N"
    tf_letter = "T" if tf_val >= 0 else "F"
    jp_letter = "J" if jp_val >= 0 else "P"

    mbti_type = ei_letter + sn_letter + tf_letter + jp_letter

    return {
        "axes": {
            "EI": ei_val,
            "SN": sn_val,
            "TF": tf_val,
            "JP": jp_val,
        },
        "type": mbti_type,
    }
