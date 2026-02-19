"""
Entry point: delegates to adaptive_quiz.experiments.run_experiments.
Run from repo root: python run_experiments.py
Or: python -m adaptive_quiz.experiments.run_experiments
"""

from adaptive_quiz.experiments.run_experiments import (
    run_single_user_simulation,
    run_multiple_users_comparison,
    compute_statistics,
    save_results,
    plot_results,
)

__all__ = [
    "run_single_user_simulation",
    "run_multiple_users_comparison",
    "compute_statistics",
    "save_results",
    "plot_results",
]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
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
