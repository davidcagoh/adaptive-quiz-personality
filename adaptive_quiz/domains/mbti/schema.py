"""
MBTI schema loader.
Returns schema compatible with AdaptiveEngine.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_mbti_schema(schema_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load MBTI schema from JSON.

    Parameters
    ----------
    schema_path : str or Path, optional
        Path to schema JSON file. Defaults to adaptive_quiz/schemas/mbti.json.

    Returns
    -------
    dict
        Schema compatible with AdaptiveEngine:
        {
            "dimensions": [{"name": str, "positive": str, "negative": str}, ...],
            "questions": [{"id": str, "text": str, "weights": [float, ...]}, ...]
        }
    """
    if schema_path is None:
        schema_path = Path(__file__).resolve().parent.parent.parent / "schemas" / "mbti.json"
    with open(schema_path) as f:
        return json.load(f)
