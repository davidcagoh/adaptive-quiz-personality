"""
Experiment script: compares selection strategies on the MBTI schema.
Uses AdaptiveEngine with SignConfidenceStopping so questions_to_convergence
reflects the actual stopping rule firing, not exhausting the question bank.
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from adaptive_quiz.simulation.synthetic_user import SyntheticUser

from adaptive_quiz.core import (
    AdaptiveEngine,
    SignConfidenceStopping,
    generate_question_weights,
)
from adaptive_quiz.domains.mbti import MBTIScorer, load_mbti_schema


def _make_random_selector(rng: np.random.RandomState) -> Callable:
    """Return a selection strategy that picks a random unasked question."""

    def selector(mu, Sigma, question_pool, asked_indices):
        remaining = [i for i in range(len(question_pool)) if i not in asked_indices]
        idx = rng.choice(remaining)
        return idx, question_pool[idx]

    return selector


def _schema_from_weights(d: int, w_list: List[np.ndarray]) -> Dict[str, Any]:
    """Build a schema dict from dimension count and list of weight vectors."""
    return {
        "name": "Synthetic",
        "dimensions": [{"name": f"dim_{i}"} for i in range(d)],
        "questions": [{"id": f"q{i}", "text": "", "weights": w.tolist()} for i, w in enumerate(w_list)],
    }


def run_single_user_simulation(
    user: SyntheticUser,
    schema: Dict[str, Any],
    d: int,
    T: int,
    selection_mode: str = "variance",
    random_seed: Optional[int] = None,
    confidence_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Run a single user simulation with AdaptiveEngine + SignConfidenceStopping.

    Returns dict with mu_traj, Sigma_traj, per_axis_variance_traj, per_axis_error_traj,
    total_error_traj, uncertainty_traj, questions_to_convergence.
    """
    true_theta = user.theta_true
    questions = schema["questions"]
    T = min(T, len(questions))
    stopping_rule = SignConfidenceStopping(confidence_threshold=confidence_threshold)

    if selection_mode == "random":
        rng = np.random.RandomState(random_seed)
        strategy = _make_random_selector(rng)
        engine = AdaptiveEngine(
            schema=schema,
            selection_strategy=strategy,
            stopping_rule=stopping_rule,
        )
    else:
        engine = AdaptiveEngine(
            schema=schema,
            selection_mode=selection_mode,
            stopping_rule=stopping_rule,
        )

    mu_traj = [engine.get_state()["mu"].copy()]
    Sigma_traj = [engine.get_state()["Sigma"].copy()]
    questions_to_convergence = T  # default if stopping rule never fires

    for step in range(T):
        next_q = engine.get_next_question()
        if next_q is None:
            questions_to_convergence = step
            break
        question_id, w_t = next_q
        y_t, r_t, _ = user.simulate_response(w_t)
        engine.submit_answer(question_id, y_t, r_t)
        state = engine.get_state()
        mu = state["mu"]
        Sigma = state["Sigma"]
        mu_traj.append(mu.copy())
        Sigma_traj.append(Sigma.copy())
        if engine.is_complete():
            questions_to_convergence = step + 1
            break

    mu_traj = np.array(mu_traj)
    Sigma_traj = np.array(Sigma_traj)
    per_axis_variance_traj = np.array([np.diag(Sigma_traj[i]) for i in range(len(Sigma_traj))])
    per_axis_error_traj = np.abs(mu_traj - true_theta)
    total_error_traj = np.linalg.norm(mu_traj - true_theta, axis=1)
    uncertainty_traj = np.array([np.trace(Sigma_traj[i]) for i in range(len(Sigma_traj))])

    return {
        "mu_traj": mu_traj,
        "Sigma_traj": Sigma_traj,
        "per_axis_variance_traj": per_axis_variance_traj,
        "per_axis_error_traj": per_axis_error_traj,
        "total_error_traj": total_error_traj,
        "uncertainty_traj": uncertainty_traj,
        "questions_to_convergence": questions_to_convergence,
    }


def run_multiple_users_comparison(
    d: int = 4,
    T: int = 80,
    num_users: int = 20,
    seed: int = 1234,
    selection_modes: List[str] = ("variance", "random"),
    schema: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Run multiple synthetic users, comparing selection strategies via AdaptiveEngine.
    If schema is None, uses the MBTI schema. Pass a synthetic schema for unit-testing.
    """
    if schema is None:
        schema = load_mbti_schema()
        T = len(schema["questions"])

    rng = np.random.RandomState(seed)
    results: Dict[str, Dict[str, List]] = {}
    for mode in selection_modes:
        results[mode] = {
            "final_uncertainty": [],
            "uncertainty_traj": [],
            "total_error_traj": [],
            "final_total_error": [],
            "per_axis_variance_traj": [],
            "per_axis_error_traj": [],
            "final_per_axis_variance": [],
            "final_per_axis_error": [],
            "questions_to_convergence": [],
        }

    for user_idx in range(num_users):
        user_seed = seed + user_idx
        user = SyntheticUser(d=d, seed=user_seed)

        for mode in selection_modes:
            sim_result = run_single_user_simulation(
                user, schema, d, T,
                selection_mode=mode,
                random_seed=user_seed + 20000 if mode == "random" else None,
                confidence_threshold=confidence_threshold,
            )
            results[mode]["final_uncertainty"].append(sim_result["uncertainty_traj"][-1])
            results[mode]["uncertainty_traj"].append(sim_result["uncertainty_traj"])
            results[mode]["total_error_traj"].append(sim_result["total_error_traj"])
            results[mode]["final_total_error"].append(sim_result["total_error_traj"][-1])
            results[mode]["per_axis_variance_traj"].append(sim_result["per_axis_variance_traj"])
            results[mode]["per_axis_error_traj"].append(sim_result["per_axis_error_traj"])
            results[mode]["final_per_axis_variance"].append(sim_result["per_axis_variance_traj"][-1])
            results[mode]["final_per_axis_error"].append(sim_result["per_axis_error_traj"][-1])
            results[mode]["questions_to_convergence"].append(sim_result["questions_to_convergence"])

    for mode in selection_modes:
        for key in ["final_uncertainty", "final_total_error", "questions_to_convergence"]:
            results[mode][key] = np.array(results[mode][key])
    return results


def compute_statistics(
    results: Dict[str, Any],
    selection_modes: List[str] = ("variance", "random"),
) -> Dict[str, Any]:
    """Statistical tests comparing each selection mode vs random."""
    stats_dict = {}
    for mode in selection_modes:
        if mode == "random":
            continue
        adaptive_unc = results[mode]["final_uncertainty"]
        random_unc = results["random"]["final_uncertainty"]
        t_unc, p_unc = stats.ttest_rel(adaptive_unc, random_unc)
        eff_unc = (np.mean(adaptive_unc) - np.mean(random_unc)) / (np.std(random_unc) + 1e-10)
        adaptive_err = results[mode]["final_total_error"]
        random_err = results["random"]["final_total_error"]
        t_err, p_err = stats.ttest_rel(adaptive_err, random_err)
        eff_err = (np.mean(adaptive_err) - np.mean(random_err)) / (np.std(random_err) + 1e-10)
        adaptive_qtc = np.array(results[mode]["questions_to_convergence"])
        random_qtc = np.array(results["random"]["questions_to_convergence"])
        t_qtc, p_qtc = stats.ttest_rel(adaptive_qtc, random_qtc)
        eff_qtc = (np.mean(adaptive_qtc) - np.mean(random_qtc)) / (np.std(random_qtc) + 1e-10)

        stats_dict[mode] = {
            "questions_to_convergence": {
                "t_statistic": float(t_qtc), "p_value": float(p_qtc), "effect_size": float(eff_qtc),
                "mean_adaptive": float(np.mean(adaptive_qtc)), "mean_random": float(np.mean(random_qtc)),
                "improvement_pct": float((1 - np.mean(adaptive_qtc) / (np.mean(random_qtc) + 1e-10)) * 100),
            },
            "uncertainty": {
                "t_statistic": float(t_unc), "p_value": float(p_unc), "effect_size": float(eff_unc),
                "mean_adaptive": float(np.mean(adaptive_unc)), "mean_random": float(np.mean(random_unc)),
                "improvement_pct": float((1 - np.mean(adaptive_unc) / (np.mean(random_unc) + 1e-10)) * 100),
            },
            "error": {
                "t_statistic": float(t_err), "p_value": float(p_err), "effect_size": float(eff_err),
                "mean_adaptive": float(np.mean(adaptive_err)), "mean_random": float(np.mean(random_err)),
                "improvement_pct": float((1 - np.mean(adaptive_err) / (np.mean(random_err) + 1e-10)) * 100),
            },
        }
    return stats_dict


def save_results(
    results: Dict[str, Any],
    stats_dict: Dict[str, Any],
    output_dir: str = "results",
    prefix: str = "experiment",
) -> None:
    """Save results to CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    summary = {"statistics": stats_dict, "summary_metrics": {}}
    for mode in results:
        summary["summary_metrics"][mode] = {
            "mean_final_uncertainty": float(np.mean(results[mode]["final_uncertainty"])),
            "std_final_uncertainty": float(np.std(results[mode]["final_uncertainty"])),
            "mean_final_error": float(np.mean(results[mode]["final_total_error"])),
            "std_final_error": float(np.std(results[mode]["final_total_error"])),
            "mean_questions_to_convergence": float(np.mean(results[mode]["questions_to_convergence"])),
        }
    with open(output_path / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    d = results[list(results.keys())[0]]["final_per_axis_variance"].shape[1]
    with open(output_path / f"{prefix}_per_user.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["user_id", "method", "questions_to_convergence", "final_uncertainty", "final_error"]
            + [f"final_variance_axis_{i}" for i in range(d)]
            + [f"final_error_axis_{i}" for i in range(d)]
        )
        for mode in results:
            for user_idx in range(len(results[mode]["final_uncertainty"])):
                row = [
                    user_idx, mode,
                    results[mode]["questions_to_convergence"][user_idx],
                    results[mode]["final_uncertainty"][user_idx],
                    results[mode]["final_total_error"][user_idx],
                ]
                row.extend(results[mode]["final_per_axis_variance"][user_idx])
                row.extend(results[mode]["final_per_axis_error"][user_idx])
                writer.writerow(row)
    print(f"Results saved to {output_path}/")


def plot_results(
    results: Dict[str, Any],
    stats_dict: Dict[str, Any],
    selection_modes: List[str] = ("variance", "random"),
) -> plt.Figure:
    """Create trajectory and distribution figures."""
    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 3, 1)
    for mode in selection_modes:
        trajs = results[mode]["uncertainty_traj"]
        max_len = max(len(t) for t in trajs)
        padded = np.array([np.pad(t, (0, max_len - len(t)), constant_values=t[-1]) for t in trajs])
        m, s = padded.mean(axis=0), padded.std(axis=0)
        ax1.plot(m, label=mode)
        ax1.fill_between(range(len(m)), m - s, m + s, alpha=0.2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Trace(Σ)")
    ax1.set_title("Uncertainty Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    for mode in selection_modes:
        trajs = results[mode]["total_error_traj"]
        max_len = max(len(t) for t in trajs)
        padded = np.array([np.pad(t, (0, max_len - len(t)), constant_values=t[-1]) for t in trajs])
        m, s = padded.mean(axis=0), padded.std(axis=0)
        ax2.plot(m, label=mode)
        ax2.fill_between(range(len(m)), m - s, m + s, alpha=0.2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("||μ - θ_true||")
    ax2.set_title("Posterior Error Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    ax3.boxplot(
        [results[mode]["questions_to_convergence"] for mode in selection_modes],
        tick_labels=selection_modes,
    )
    ax3.set_ylabel("Questions to convergence")
    ax3.set_title("Questions to Convergence")
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = plt.subplot(2, 3, 4)
    for mode in selection_modes:
        q = results[mode]["questions_to_convergence"]
        ax4.hist(q, bins=20, alpha=0.6, label=mode)
    ax4.set_xlabel("Questions asked")
    ax4.set_ylabel("Count")
    ax4.set_title("Distribution of Questions Asked")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    ax5.boxplot(
        [results[mode]["final_uncertainty"] for mode in selection_modes],
        tick_labels=selection_modes,
    )
    ax5.set_ylabel("Final Trace(Σ)")
    ax5.set_title("Final Uncertainty Distribution")
    ax5.grid(True, alpha=0.3, axis="y")

    ax6 = plt.subplot(2, 3, 6)
    ax6.boxplot(
        [results[mode]["final_total_error"] for mode in selection_modes],
        tick_labels=selection_modes,
    )
    ax6.set_ylabel("Final ||μ - θ_true||")
    ax6.set_title("Final Error Distribution")
    ax6.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    num_users = 200
    seed = 42
    selection_modes = ["variance", "random"]

    print("Running experiment on MBTI schema with SignConfidenceStopping(0.85)...")
    results = run_multiple_users_comparison(
        num_users=num_users,
        seed=seed,
        selection_modes=selection_modes,
    )
    stats_dict = compute_statistics(results, selection_modes)

    print("\n=== Statistical Comparison ===")
    for mode in stats_dict:
        qtc = stats_dict[mode]["questions_to_convergence"]
        print(f"\n{mode.upper()} vs RANDOM:")
        print(f"  ** Questions to convergence (HEADLINE METRIC):")
        print(f"     Adaptive avg: {qtc['mean_adaptive']:.1f} questions  |  "
              f"Random avg: {qtc['mean_random']:.1f} questions  |  "
              f"Reduction: {-qtc['improvement_pct']:.1f}%  |  "
              f"p={qtc['p_value']:.4f}, d={qtc['effect_size']:.3f}")
        print(f"  Uncertainty: p={stats_dict[mode]['uncertainty']['p_value']:.4f}, "
              f"effect={stats_dict[mode]['uncertainty']['effect_size']:.3f}, "
              f"improvement={stats_dict[mode]['uncertainty']['improvement_pct']:.1f}%")
        print(f"  Error: p={stats_dict[mode]['error']['p_value']:.4f}, "
              f"effect={stats_dict[mode]['error']['effect_size']:.3f}, "
              f"improvement={stats_dict[mode]['error']['improvement_pct']:.1f}%")

    save_results(results, stats_dict, output_dir="results", prefix="experiment")
    fig = plot_results(results, stats_dict, selection_modes)
    fig.savefig("results/experiment_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
