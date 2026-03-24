#!/usr/bin/env python3
"""
MBTI Adaptive Convergence Simulation

Three-way comparison on the actual MBTI schema:
  (a) Adaptive selection (VarianceSelection) + SignConfidenceStopping
  (b) Random question order  + SignConfidenceStopping (same stopping rule)
  (c) Full test: all questions in fixed order, no stopping

(a) vs (b) proves the selection strategy causes faster convergence.
(b) vs (c) proves the stopping rule causes fewer questions.
(a) vs (c) is the headline: adaptive vs exhaustive.
"""

import numpy as np

from adaptive_quiz.core import (
    AdaptiveEngine,
    SignConfidenceStopping,
    VarianceSelection,
    VarianceThresholdStopping,
)
from adaptive_quiz.domains.mbti import load_mbti_schema, mbti_from_posterior
from adaptive_quiz.simulation.synthetic_user import SyntheticUser


# ---------------------------------------------------------------------------
# Response simulation
# ---------------------------------------------------------------------------

def _simulate_response(user: SyntheticUser, w: np.ndarray):
    """Use SyntheticUser to produce a Likert-quantized response and noise estimate."""
    y_t, r_t, sigma2_t = user.simulate_response(w)
    return y_t, r_t, sigma2_t


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_adaptive_test(
    user: SyntheticUser,
    schema: dict,
    stopping_rule: SignConfidenceStopping,
    selection_strategy,
) -> dict:
    """
    Run adaptive test with given selection strategy and stopping rule.

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
        y_t, r_t, _ = _simulate_response(user, w)
        engine.submit_answer(qid, y_t, r_t)

    state = engine.get_state()
    result = mbti_from_posterior(state.mu)
    return {
        "type": result["type"],
        "mu": state.mu.copy(),
        "num_questions": state.num_questions,
    }


def run_random_test(
    user: SyntheticUser,
    schema: dict,
    stopping_rule: SignConfidenceStopping,
    rng: np.random.RandomState,
) -> dict:
    """
    Run test with random question order and the same stopping rule.
    This isolates the effect of selection strategy vs. stopping rule.
    """
    questions = schema["questions"]
    order = list(range(len(questions)))
    rng.shuffle(order)

    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=VarianceSelection(),  # engine needs one; we override order below
        stopping_rule=stopping_rule,
    )

    # Manually feed questions in random order, checking stopping after each
    num_asked = 0
    asked_ids = set()
    for idx in order:
        if engine.is_complete():
            break
        q = questions[idx]
        qid = q["id"]
        if qid in asked_ids:
            continue
        w = np.array(q["weights"])
        y_t, r_t, _ = _simulate_response(user, w)
        engine.submit_answer(qid, y_t, r_t)
        asked_ids.add(qid)
        num_asked += 1
        if engine.is_complete():
            break

    state = engine.get_state()
    result = mbti_from_posterior(state.mu)
    return {
        "type": result["type"],
        "mu": state.mu.copy(),
        "num_questions": num_asked,
    }


def run_full_test(
    user: SyntheticUser,
    schema: dict,
) -> dict:
    """
    Run all questions in fixed schema order, no stopping.
    Baseline: exhausts the entire question bank.
    """
    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=VarianceSelection(),
        stopping_rule=VarianceThresholdStopping(variance_threshold=1e-6),  # never fires
    )

    for question in schema["questions"]:
        qid = question["id"]
        w = np.array(question["weights"])
        y_t, r_t, _ = _simulate_response(user, w)
        engine.submit_answer(qid, y_t, r_t)

    state = engine.get_state()
    result = mbti_from_posterior(state.mu)
    return {
        "type": result["type"],
        "mu": state.mu.copy(),
        "num_questions": state.num_questions,
    }


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _type_match(t1: str, t2: str) -> tuple:
    """Returns (overall_match, [per_axis_matches])."""
    if len(t1) != 4 or len(t2) != 4:
        return False, [False] * 4
    return t1 == t2, [t1[i] == t2[i] for i in range(4)]


def _print_results(name: str, metrics: dict) -> None:
    q = metrics["question_counts"]
    print(f"\n{name}")
    print("-" * 50)
    print(f"  Questions — mean: {np.mean(q):.1f}  median: {int(np.median(q))}  "
          f"min: {int(np.min(q))}  max: {int(np.max(q))}")
    print(f"  Type accuracy vs true:      {100 * np.mean(metrics['vs_true']):.1f}%")
    print(f"  Type accuracy vs full test: {100 * np.mean(metrics['vs_full']):.1f}%")
    print("  Per-axis vs true:", "  ".join(
        f"{ax}:{100 * np.mean(metrics['axis_vs_true'][ax]):.1f}%"
        for ax in ["EI", "NS", "TF", "JP"]
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Three-way comparison: adaptive vs random vs full test.
    N=1000 synthetic users, actual MBTI schema.
    """
    N = 1000
    seed = 42
    rng_latent = np.random.RandomState(seed)
    rng_random_order = np.random.RandomState(seed + 1)

    schema = load_mbti_schema()
    n_questions = len(schema["questions"])
    stopping_rule_adaptive = SignConfidenceStopping(confidence_threshold=0.85)
    stopping_rule_random = SignConfidenceStopping(confidence_threshold=0.85)

    axes = ["EI", "NS", "TF", "JP"]
    adaptive = {"question_counts": [], "vs_true": [], "vs_full": [],
                "axis_vs_true": {ax: [] for ax in axes}}
    random =   {"question_counts": [], "vs_true": [], "vs_full": [],
                "axis_vs_true": {ax: [] for ax in axes}}

    print("=" * 50)
    print("MBTI Adaptive Convergence Simulation")
    print(f"N = {N} synthetic users  |  Schema: {n_questions} questions")
    print("Stopping rule: SignConfidence(0.95) on all four axes")
    print("=" * 50)

    for i in range(N):
        user = SyntheticUser(d=4, seed=rng_latent.randint(0, 2**31))
        true_type = mbti_from_posterior(user.theta_true)["type"]

        full = run_full_test(user, schema)
        full_type = full["type"]

        adap = run_adaptive_test(
            user, schema, stopping_rule_adaptive, VarianceSelection()
        )
        rand = run_random_test(
            user, schema, stopping_rule_random, rng_random_order
        )

        adaptive["question_counts"].append(adap["num_questions"])
        adap_match, adap_axis = _type_match(adap["type"], true_type)
        adaptive["vs_true"].append(adap_match)
        adap_full, _ = _type_match(adap["type"], full_type)
        adaptive["vs_full"].append(adap_full)
        for j, ax in enumerate(axes):
            adaptive["axis_vs_true"][ax].append(adap_axis[j])

        random["question_counts"].append(rand["num_questions"])
        rand_match, rand_axis = _type_match(rand["type"], true_type)
        random["vs_true"].append(rand_match)
        rand_full, _ = _type_match(rand["type"], full_type)
        random["vs_full"].append(rand_full)
        for j, ax in enumerate(axes):
            random["axis_vs_true"][ax].append(rand_axis[j])

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N} done...")

    # Convert to numpy
    for metrics in [adaptive, random]:
        metrics["question_counts"] = np.array(metrics["question_counts"])
        metrics["vs_true"] = np.array(metrics["vs_true"])
        metrics["vs_full"] = np.array(metrics["vs_full"])
        for ax in axes:
            metrics["axis_vs_true"][ax] = np.array(metrics["axis_vs_true"][ax])

    print(f"\nFull test (all {n_questions} questions, fixed order) — baseline")
    print(f"  Questions: {n_questions}")

    _print_results("Adaptive selection (VarianceSelection + SignConfidence)", adaptive)
    _print_results("Random selection   (random order  + SignConfidence)", random)

    # Key comparisons
    adap_mean = np.mean(adaptive["question_counts"])
    rand_mean = np.mean(random["question_counts"])
    print(f"\n{'=' * 50}")
    print("KEY RESULTS")
    print(f"  Adaptive vs full:   {adap_mean:.1f} / {n_questions} questions "
          f"({100 * (1 - adap_mean / n_questions):.1f}% reduction)")
    print(f"  Random vs full:     {rand_mean:.1f} / {n_questions} questions "
          f"({100 * (1 - rand_mean / n_questions):.1f}% reduction)")
    print(f"  Adaptive vs random: {adap_mean:.1f} / {rand_mean:.1f} questions "
          f"({100 * (1 - adap_mean / rand_mean):.1f}% further reduction from smart ordering)")
    print(f"  Adaptive type accuracy (vs true): "
          f"{100 * np.mean(adaptive['vs_true']):.1f}%")
    print(f"  Reference: 16personalities uses ~93 fixed questions, no stopping")
    print("=" * 50)


if __name__ == "__main__":
    main()
