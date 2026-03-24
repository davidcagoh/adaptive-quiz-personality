"""MBTI domain adapter: schema and scoring."""

from adaptive_quiz.domains.mbti.schema import load_mbti_schema
from adaptive_quiz.domains.mbti.scoring import mbti_from_posterior, MBTIScorer

__all__ = ["load_mbti_schema", "mbti_from_posterior", "MBTIScorer"]
