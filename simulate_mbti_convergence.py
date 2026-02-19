#!/usr/bin/env python3
"""
MBTI Adaptive Convergence Simulation

Compares selection strategies (VarianceSelection vs InformationGainSelection)
under SignConfidenceStopping(0.95).
"""

import numpy as np

from adaptive_quiz.core import (
    AdaptiveEngine,
    InformationGainSelection,
    SignConfidenceStopping,
    VarianceSelection,
    VarianceThresholdStopping,
)
from adaptive_quiz.domains.mbti import load_mbti_schema, mbti_from_posterior


def simulate_response(mu_true: np.ndarray, w: np.ndarray, noise_sigma: float = np.sqrt(0.2)) -> float:
    """
    Simulate observed response given true latent vector and question weight.

    Parameters
    ----------
    mu_true : np.ndarray
        True latent personality vector (4,).
    w : np.ndarray
        Question weight vector (4,).
    noise_sigma : float
        Standard deviation of observation noise.

    Returns
    -------
    float
        Observed response: w^T mu_true + noise
    """
    r_true = w @ mu_true
    noise = np.random.randn() * noise_sigma
    return r_true + noise


def run_full_test(mu_true: np.ndarray, schema: dict, noise_sigma: float = np.sqrt(0.2)) -> dict:
    """
    Run full 30-question test (baseline).

    Parameters
    ----------
    mu_true : np.ndarray
        True latent personality vector (4,).
    schema : dict
        MBTI schema with questions.
    noise_sigma : float
        Standard deviation of observation noise.

    Returns
    -------
    dict
        {"type": str, "mu": np.ndarray}
    """
    # Create engine with no stopping (very low threshold so it never stops)
    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=None,  # Not used since we answer all questions directly
        stopping_rule=VarianceThresholdStopping(variance_threshold=0.001),  # Effectively no stopping
        selection_mode="variance",
    )

    questions = schema["questions"]

    # Answer all 30 questions in fixed order
    for question in questions:
        qid = question["id"]
        w = np.array(question["weights"])
        r_obs = simulate_response(mu_true, w, noise_sigma)
        engine.submit_answer(qid, r_obs, 1.0)

    state = engine.get_state()
    result = mbti_from_posterior(state.mu)
    return {"type": result["type"], "mu": state.mu.copy()}


def run_adaptive_test(
    mu_true: np.ndarray,
    schema: dict,
    stopping_rule,
    selection_strategy,
    noise_sigma: float = np.sqrt(0.2),
) -> dict:
    """
    Run adaptive test with given stopping rule and selection strategy.

    Parameters
    ----------
    mu_true : np.ndarray
        True latent personality vector (4,).
    schema : dict
        MBTI schema with questions.
    stopping_rule
        Stopping rule instance.
    selection_strategy
        Selection strategy instance (e.g. VarianceSelection, InformationGainSelection).
    noise_sigma : float
        Standard deviation of observation noise.

    Returns
    -------
    dict
        {"type": str, "mu": np.ndarray, "num_questions": int}
    """
    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=selection_strategy,
        stopping_rule=stopping_rule,
    )

    while not engine.is_complete():
        next_q = engine.get_next_question()
        if next_q is None:
            break

        qid, w = next_q
        r_obs = simulate_response(mu_true, w, noise_sigma)
        engine.submit_answer(qid, r_obs, 1.0)

    state = engine.get_state()
    result = mbti_from_posterior(state.mu)
    return {
        "type": result["type"],
        "mu": state.mu.copy(),
        "num_questions": state.num_questions,
    }


def compare_types(type1: str, type2: str) -> tuple[bool, list[bool]]:
    """
    Compare two MBTI types.

    Parameters
    ----------
    type1 : str
        First MBTI type (e.g., "INTJ").
    type2 : str
        Second MBTI type (e.g., "INTJ").

    Returns
    -------
    tuple[bool, list[bool]]
        (overall_match, [ei_match, sn_match, tf_match, jp_match])
    """
    if len(type1) != 4 or len(type2) != 4:
        return False, [False] * 4

    overall = type1 == type2
    ei_match = type1[0] == type2[0]
    sn_match = type1[1] == type2[1]
    tf_match = type1[2] == type2[2]
    jp_match = type1[3] == type2[3]

    return overall, [ei_match, sn_match, tf_match, jp_match]


def run_selection_comparison(
    mu_true_samples: list,
    schema: dict,
    stopping_rule,
    noise_sigma: float,
) -> tuple[dict, dict, float]:
    """
    Run simulation comparing VarianceSelection vs InformationGainSelection.

    Uses same stopping rule and same mu_true draws for both strategies.
    Full-test is run once per mu_true; full_vs_true is computed once.

    Returns
    -------
    results_variance : dict
        Metrics for VarianceSelection.
    results_info_gain : dict
        Metrics for InformationGainSelection.
    full_vs_true_rate : float
        Full test vs TRUE agreement rate (same for both strategies).
    """
    variance = {
        "question_counts": [],
        "adaptive_vs_true": [],
        "adaptive_vs_full": [],
        "axis_vs_true": {"EI": [], "SN": [], "TF": [], "JP": []},
    }
    info_gain = {
        "question_counts": [],
        "adaptive_vs_true": [],
        "adaptive_vs_full": [],
        "axis_vs_true": {"EI": [], "SN": [], "TF": [], "JP": []},
    }
    full_vs_true_list = []

    for mu_true in mu_true_samples:
        true_type = mbti_from_posterior(mu_true)["type"]
        full_result = run_full_test(mu_true, schema, noise_sigma)
        full_type = full_result["type"]

        full_true_match, _ = compare_types(full_type, true_type)
        full_vs_true_list.append(full_true_match)

        # VarianceSelection
        adaptive_var = run_adaptive_test(
            mu_true, schema, stopping_rule, VarianceSelection(), noise_sigma
        )
        av_true, av_true_axis = compare_types(adaptive_var["type"], true_type)
        av_full, _ = compare_types(adaptive_var["type"], full_type)
        variance["question_counts"].append(adaptive_var["num_questions"])
        variance["adaptive_vs_true"].append(av_true)
        variance["adaptive_vs_full"].append(av_full)
        for i, axis in enumerate(["EI", "SN", "TF", "JP"]):
            variance["axis_vs_true"][axis].append(av_true_axis[i])

        # InformationGainSelection
        adaptive_ig = run_adaptive_test(
            mu_true, schema, stopping_rule, InformationGainSelection(), noise_sigma
        )
        ag_true, ag_true_axis = compare_types(adaptive_ig["type"], true_type)
        ag_full, _ = compare_types(adaptive_ig["type"], full_type)
        info_gain["question_counts"].append(adaptive_ig["num_questions"])
        info_gain["adaptive_vs_true"].append(ag_true)
        info_gain["adaptive_vs_full"].append(ag_full)
        for i, axis in enumerate(["EI", "SN", "TF", "JP"]):
            info_gain["axis_vs_true"][axis].append(ag_true_axis[i])

    def to_arrays(d: dict) -> dict:
        out = {
            "question_counts": np.array(d["question_counts"]),
            "adaptive_vs_true": np.array(d["adaptive_vs_true"]),
            "adaptive_vs_full": np.array(d["adaptive_vs_full"]),
            "axis_vs_true": {k: np.array(v) for k, v in d["axis_vs_true"].items()},
        }
        return out

    full_vs_true_rate = 100 * np.mean(full_vs_true_list)
    return to_arrays(variance), to_arrays(info_gain), full_vs_true_rate


def print_strategy_results(strategy_name: str, metrics: dict) -> None:
    """Print formatted results for a selection strategy."""
    print(f"\nSelection Strategy: {strategy_name}")
    print("-" * 50)
    print(f"Average questions asked: {np.mean(metrics['question_counts']):.2f}")
    print(f"Median questions asked: {int(np.median(metrics['question_counts']))}")
    print(f"Min questions asked: {int(np.min(metrics['question_counts']))}")
    print(f"Max questions asked: {int(np.max(metrics['question_counts']))}")
    print()
    print(f"Adaptive vs TRUE: {100 * np.mean(metrics['adaptive_vs_true']):.1f}%")
    print(f"Adaptive vs FULL: {100 * np.mean(metrics['adaptive_vs_full']):.1f}%")
    print("\nPer-axis agreement vs TRUE:")
    for axis in ["EI", "SN", "TF", "JP"]:
        rate = 100 * np.mean(metrics["axis_vs_true"][axis])
        print(f"{axis}: {rate:.1f}%")


def main() -> None:
    """Compare VarianceSelection vs InformationGainSelection under SignConfidenceStopping(0.95)."""
    np.random.seed(42)
    schema = load_mbti_schema()
    N = 1000
    noise_sigma = np.sqrt(0.2)
    stopping_rule = SignConfidenceStopping(confidence_threshold=0.95)

    print("=" * 50)
    print("MBTI Selection Strategy Comparison")
    print("Stopping Rule: SignConfidence (0.95)")
    print(f"Runs: {N}")
    print("=" * 50)

    mu_true_samples = [np.random.randn(4) for _ in range(N)]

    results_variance, results_info_gain, full_vs_true_rate = run_selection_comparison(
        mu_true_samples, schema, stopping_rule, noise_sigma
    )

    print("\nFull Test vs TRUE:")
    print(f"Type agreement: {full_vs_true_rate:.1f}%")

    print_strategy_results("VarianceSelection", results_variance)
    print_strategy_results("InformationGainSelection", results_info_gain)
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
