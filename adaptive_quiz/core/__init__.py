"""Core adaptive inference engine (domain-agnostic)."""

from adaptive_quiz.core.bayesian import bayesian_update
from adaptive_quiz.core.selection import (
    InformationGainSelection,
    SelectionStrategy,
    VarianceSelection,
    expected_information_gain,
    generate_question_weights,
    select_next_question,
)
from adaptive_quiz.core.stopping import (
    SignConfidenceStopping,
    StoppingRule,
    VarianceThresholdStopping,
)
from adaptive_quiz.core.engine import (
    AdaptiveEngine,
    DefaultLikertNoiseModel,
    EngineState,
    NoiseModel,
)

__all__ = [
    "bayesian_update",
    "select_next_question",
    "expected_information_gain",
    "generate_question_weights",
    "VarianceThresholdStopping",
    "SignConfidenceStopping",
    "StoppingRule",
    "AdaptiveEngine",
    "EngineState",
    "SelectionStrategy",
    "VarianceSelection",
    "InformationGainSelection",
    "NoiseModel",
    "DefaultLikertNoiseModel",
]
