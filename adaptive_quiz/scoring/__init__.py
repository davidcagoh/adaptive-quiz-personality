"""Questionnaire-specific scoring adapters."""

from adaptive_quiz.scoring.mbti_scoring import MBTIScorer, load_mbti_schema

__all__ = ["MBTIScorer", "load_mbti_schema"]
