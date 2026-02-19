# Backward compatibility: delegate to refactored core.
from adaptive_quiz.core.bayesian import bayesian_update

__all__ = ["bayesian_update"]
