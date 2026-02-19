"""
Entry point: delegates to adaptive_quiz.simulation.run_simulation.
Run from repo root: python run_simulation.py
Or: python -m adaptive_quiz.simulation.run_simulation
"""

from adaptive_quiz.simulation.run_simulation import (
    run_simulation,
    compute_metrics,
    animate_simulation,
    animate_simulation_2d,
    animate_radar,
    animate_parallel_coords,
    animate_multi_2d,
    plot_radar,
)

__all__ = [
    "run_simulation",
    "compute_metrics",
    "animate_simulation",
    "animate_simulation_2d",
    "animate_radar",
    "animate_parallel_coords",
    "animate_multi_2d",
    "plot_radar",
]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    d = 4
    T = 30
    seed = 42
    selection_mode = "variance"
    viz_mode = "radar"
    use_synthetic_pool = True
    mu_traj, Sigma_traj, responses, response_times, w_list, user = run_simulation(
        d=d, T=T, seed=seed, selection_mode=selection_mode, use_synthetic_pool=use_synthetic_pool
    )
    if not use_synthetic_pool:
        from adaptive_quiz.scoring import MBTIScorer
        scorer = MBTIScorer()
        result = scorer.score(mu_traj[-1], Sigma_traj[-1])
        print("MBTI result:", result["type"], "axis_scores:", result["axis_scores"])
    colors = plt.cm.viridis(np.linspace(0, 1, T))
    ani = animate_simulation(
        mu_traj, Sigma_traj, T, colors, mode=viz_mode,
        true_theta=user.theta_true, show_metrics=True, show_step_size=True
    )
    plt.show()
