#!/usr/bin/env python3
"""
Confidence threshold sensitivity analysis.

Sweeps SignConfidenceStopping threshold from 0.75 to 0.99 and reports
the accuracy–efficiency tradeoff: questions asked vs. type accuracy.

Run:
    python -m adaptive_quiz.experiments.threshold_sweep
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from adaptive_quiz.core import AdaptiveEngine, SignConfidenceStopping, VarianceSelection
from adaptive_quiz.domains.mbti import load_mbti_schema, mbti_from_posterior
from adaptive_quiz.simulation.synthetic_user import SyntheticUser


def run_sweep(
    thresholds: list[float],
    n_users: int = 500,
    seed: int = 42,
) -> dict[float, dict]:
    """
    For each threshold, run N synthetic users with adaptive selection and
    return mean questions asked + type accuracy vs. the full-test baseline.
    """
    schema = load_mbti_schema()
    rng = np.random.RandomState(seed)

    # Pre-generate user seeds so each threshold sees identical users
    user_seeds = [rng.randint(0, 2**31) for _ in range(n_users)]

    # Full-test baseline (run once — threshold-independent)
    print("Computing full-test baseline...")
    full_types = []
    for us in user_seeds:
        user = SyntheticUser(d=4, seed=us)
        engine = AdaptiveEngine(
            schema=schema,
            selection_strategy=VarianceSelection(),
            stopping_rule=SignConfidenceStopping(confidence_threshold=0.9999),
        )
        for q in schema["questions"]:
            y, r, _ = user.simulate_response(np.array(q["weights"]))
            engine.submit_answer(q["id"], y, r)
        full_types.append(mbti_from_posterior(engine.get_state()["mu"])["type"])

    results = {}
    for threshold in thresholds:
        print(f"  threshold={threshold:.2f} ...", end="", flush=True)
        q_counts = []
        correct_vs_full = []

        for i, us in enumerate(user_seeds):
            user = SyntheticUser(d=4, seed=us)
            engine = AdaptiveEngine(
                schema=schema,
                selection_strategy=VarianceSelection(),
                stopping_rule=SignConfidenceStopping(confidence_threshold=threshold),
            )
            while not engine.is_complete():
                nq = engine.get_next_question()
                if nq is None:
                    break
                qid, w = nq
                y, r, _ = user.simulate_response(w)
                engine.submit_answer(qid, y, r)

            state = engine.get_state()
            pred_type = mbti_from_posterior(state["mu"])["type"]
            q_counts.append(state["num_questions"])
            correct_vs_full.append(pred_type == full_types[i])

        results[threshold] = {
            "mean_questions": float(np.mean(q_counts)),
            "median_questions": int(np.median(q_counts)),
            "p10_questions": int(np.percentile(q_counts, 10)),
            "p90_questions": int(np.percentile(q_counts, 90)),
            "accuracy_vs_full": float(np.mean(correct_vs_full)),
            "q_counts": q_counts,
        }
        print(f" {results[threshold]['mean_questions']:.1f}q  acc={results[threshold]['accuracy_vs_full']:.3f}")

    return results


def print_table(results: dict[float, dict]) -> None:
    print("\nThreshold | Mean Q | Median Q | P10–P90     | Accuracy vs full")
    print("-" * 62)
    for t in sorted(results):
        r = results[t]
        print(
            f"  {t:.2f}    |  {r['mean_questions']:5.1f} |    {r['median_questions']:3d}   | "
            f"{r['p10_questions']:2d}–{r['p90_questions']:2d}        | {100*r['accuracy_vs_full']:.1f}%"
        )


def plot_pareto(results: dict[float, dict], output_path: str = "results/threshold_sweep.png") -> None:
    thresholds = sorted(results)
    mean_q = [results[t]["mean_questions"] for t in thresholds]
    accuracy = [100 * results[t]["accuracy_vs_full"] for t in thresholds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(thresholds, mean_q, "o-", color="steelblue")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean questions asked")
    ax.set_title("Questions asked vs. threshold")
    ax.grid(True, alpha=0.3)
    for t, q in zip(thresholds, mean_q):
        ax.annotate(f"{q:.1f}", (t, q), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax = axes[1]
    ax.plot(mean_q, accuracy, "o-", color="darkorange")
    for t, q, a in zip(thresholds, mean_q, accuracy):
        ax.annotate(f"  {t:.2f}", (q, a), fontsize=8)
    ax.set_xlabel("Mean questions asked")
    ax.set_ylabel("Type accuracy vs. full test (%)")
    ax.set_title("Accuracy–Efficiency Pareto curve")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    thresholds = [0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    n_users = 500

    print(f"Threshold sweep — N={n_users} synthetic users, MBTI schema")
    print("=" * 62)

    results = run_sweep(thresholds, n_users=n_users)
    print_table(results)
    plot_pareto(results)

    # Save raw numbers
    out = {str(t): {k: v for k, v in r.items() if k != "q_counts"} for t, r in results.items()}
    Path("results").mkdir(exist_ok=True)
    with open("results/threshold_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Raw results saved to results/threshold_sweep.json")
