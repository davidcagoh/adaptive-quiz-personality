"""
Experiment script: uses AdaptiveEngine, synthetic or MBTI schema, MBTIScorer only at final stage.
Plotting and statistics unchanged.
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import sys
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from synthetic_user import SyntheticUser

from adaptive_quiz.core import AdaptiveEngine, generate_question_weights
from adaptive_quiz.scoring import MBTIScorer


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
) -> Dict[str, Any]:
    """
    Run a single user simulation with AdaptiveEngine.

    Returns dict with mu_traj, Sigma_traj, per_axis_variance_traj, per_axis_error_traj,
    total_error_traj, uncertainty_traj.
    """
    true_theta = user.theta_true
    questions = schema["questions"]
    T = min(T, len(questions))

    if selection_mode == "random":
        rng = np.random.RandomState(random_seed)
        strategy = _make_random_selector(rng)
        engine = AdaptiveEngine(schema=schema, selection_strategy=strategy)
    else:
        engine = AdaptiveEngine(schema=schema, selection_mode=selection_mode)

    mu_traj = [engine.get_state()["mu"].copy()]
    Sigma_traj = [engine.get_state()["Sigma"].copy()]

    for _ in range(T):
        next_q = engine.get_next_question()
        if next_q is None:
            break
        question_id, w_t = next_q
        y_t, r_t, _ = user.simulate_response(w_t)
        engine.submit_answer(question_id, y_t, r_t)
        state = engine.get_state()
        mu = state["mu"]
        Sigma = state["Sigma"]
        mu_traj.append(mu.copy())
        Sigma_traj.append(Sigma.copy())

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
    }


def run_multiple_users_comparison(
    d: int = 4,
    T: int = 30,
    num_users: int = 20,
    seed: int = 1234,
    selection_modes: List[str] = ("variance", "random"),
) -> Dict[str, Any]:
    """
    Run multiple synthetic users, comparing selection strategies via AdaptiveEngine.
    Uses synthetic question pool (same for all modes per user).
    """
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
        }

    for user_idx in range(num_users):
        user_seed = seed + user_idx
        user = SyntheticUser(d=d, seed=user_seed)
        w_list = generate_question_weights(d, T, random_state=np.random.RandomState(user_seed + 10000))
        schema = _schema_from_weights(d, w_list)

        for mode in selection_modes:
            sim_result = run_single_user_simulation(
                user, schema, d, T,
                selection_mode=mode,
                random_seed=user_seed + 20000 if mode == "random" else None,
            )
            results[mode]["final_uncertainty"].append(sim_result["uncertainty_traj"][-1])
            results[mode]["uncertainty_traj"].append(sim_result["uncertainty_traj"])
            results[mode]["total_error_traj"].append(sim_result["total_error_traj"])
            results[mode]["final_total_error"].append(sim_result["total_error_traj"][-1])
            results[mode]["per_axis_variance_traj"].append(sim_result["per_axis_variance_traj"])
            results[mode]["per_axis_error_traj"].append(sim_result["per_axis_error_traj"])
            results[mode]["final_per_axis_variance"].append(sim_result["per_axis_variance_traj"][-1])
            results[mode]["final_per_axis_error"].append(sim_result["per_axis_error_traj"][-1])

    for mode in selection_modes:
        for key in results[mode]:
            results[mode][key] = np.array(results[mode][key])
    return results


def compute_statistics(
    results: Dict[str, Any],
    selection_modes: List[str] = ("variance", "random"),
) -> Dict[str, Any]:
    """Statistical tests comparing selection modes vs random."""
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
        initial_unc = results[mode]["uncertainty_traj"][:, 0]
        target_unc = 0.5 * initial_unc
        conv_steps = []
        for i, traj in enumerate(results[mode]["uncertainty_traj"]):
            below = np.where(traj <= target_unc[i])[0]
            conv_steps.append(below[0] if len(below) > 0 else len(traj) - 1)
        random_conv = []
        for i, traj in enumerate(results["random"]["uncertainty_traj"]):
            below = np.where(traj <= target_unc[i])[0]
            random_conv.append(below[0] if len(below) > 0 else len(traj) - 1)
        t_conv, p_conv = stats.ttest_rel(conv_steps, random_conv)
        eff_conv = (np.mean(conv_steps) - np.mean(random_conv)) / (np.std(random_conv) + 1e-10)
        stats_dict[mode] = {
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
            "convergence": {
                "t_statistic": float(t_conv), "p_value": float(p_conv), "effect_size": float(eff_conv),
                "mean_adaptive": float(np.mean(conv_steps)), "mean_random": float(np.mean(random_conv)),
                "improvement_pct": float((1 - np.mean(conv_steps) / (np.mean(random_conv) + 1e-10)) * 100),
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
        }
    with open(output_path / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    d = results[list(results.keys())[0]]["final_per_axis_variance"].shape[1]
    with open(output_path / f"{prefix}_per_user.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["user_id", "method", "final_uncertainty", "final_error"]
            + [f"final_variance_axis_{i}" for i in range(d)]
            + [f"final_error_axis_{i}" for i in range(d)]
        )
        for mode in results:
            for user_idx in range(len(results[mode]["final_uncertainty"])):
                row = [
                    user_idx, mode,
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
    """Create trajectory and boxplot figures."""
    fig = plt.figure(figsize=(16, 10))
    ax1 = plt.subplot(2, 3, 1)
    for mode in selection_modes:
        m = results[mode]["uncertainty_traj"].mean(axis=0)
        s = results[mode]["uncertainty_traj"].std(axis=0)
        ax1.plot(m, label=mode)
        ax1.fill_between(range(len(m)), m - s, m + s, alpha=0.2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Trace of Posterior Covariance")
    ax1.set_title("Uncertainty Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    for mode in selection_modes:
        m = results[mode]["total_error_traj"].mean(axis=0)
        s = results[mode]["total_error_traj"].std(axis=0)
        ax2.plot(m, label=mode)
        ax2.fill_between(range(len(m)), m - s, m + s, alpha=0.2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("||mu - theta_true||")
    ax2.set_title("Posterior Error Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    d = results[selection_modes[0]]["per_axis_variance_traj"].shape[2]
    ax3 = plt.subplot(2, 3, 3)
    for axis in range(d):
        for mode in selection_modes:
            m = results[mode]["per_axis_variance_traj"][:, :, axis].mean(axis=0)
            ax3.plot(m, label=f"{mode} axis {axis}", linestyle="--" if mode == "random" else "-", alpha=0.7)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Per-Axis Variance")
    ax3.set_title("Per-Axis Uncertainty")
    ax3.legend(ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    for axis in range(d):
        for mode in selection_modes:
            m = results[mode]["per_axis_error_traj"][:, :, axis].mean(axis=0)
            ax4.plot(m, label=f"{mode} axis {axis}", linestyle="--" if mode == "random" else "-", alpha=0.7)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Per-Axis Error")
    ax4.set_title("Per-Axis Error")
    ax4.legend(ncol=2, fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    ax5.boxplot([results[mode]["final_uncertainty"] for mode in selection_modes], tick_labels=selection_modes)
    ax5.set_ylabel("Final Uncertainty")
    ax5.set_title("Final Uncertainty Distribution")
    ax5.grid(True, alpha=0.3, axis="y")

    ax6 = plt.subplot(2, 3, 6)
    ax6.boxplot([results[mode]["final_total_error"] for mode in selection_modes], tick_labels=selection_modes)
    ax6.set_ylabel("Final Error")
    ax6.set_title("Final Error Distribution")
    ax6.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    d = 4
    T = 30
    num_users = 20
    seed = 42
    selection_modes = ["variance", "random", "info_gain"]

    results = run_multiple_users_comparison(
        d=d, T=T, num_users=num_users, seed=seed, selection_modes=selection_modes
    )
    stats_dict = compute_statistics(results, selection_modes)

    print("\n=== Statistical Comparison ===")
    for mode in stats_dict:
        print(f"\n{mode.upper()} vs RANDOM:")
        print(f"  Uncertainty: p={stats_dict[mode]['uncertainty']['p_value']:.4f}, "
              f"effect={stats_dict[mode]['uncertainty']['effect_size']:.3f}, "
              f"improvement={stats_dict[mode]['uncertainty']['improvement_pct']:.1f}%")
        print(f"  Error: p={stats_dict[mode]['error']['p_value']:.4f}, "
              f"effect={stats_dict[mode]['error']['effect_size']:.3f}, "
              f"improvement={stats_dict[mode]['error']['improvement_pct']:.1f}%")
        print(f"  Convergence: p={stats_dict[mode]['convergence']['p_value']:.4f}, "
              f"effect={stats_dict[mode]['convergence']['effect_size']:.3f}, "
              f"improvement={stats_dict[mode]['convergence']['improvement_pct']:.1f}%")

    save_results(results, stats_dict, output_dir="results", prefix="experiment")
    fig = plot_results(results, stats_dict, selection_modes)
    fig.savefig("results/experiment_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
