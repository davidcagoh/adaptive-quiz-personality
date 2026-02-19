# Backward compatibility: delegate to refactored core.
from adaptive_quiz.core.selection import (
    generate_question_weights,
    select_next_question,
    expected_information_gain,
)

__all__ = ["generate_question_weights", "select_next_question", "expected_information_gain"]
