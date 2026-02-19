"""
Simulation script: uses AdaptiveEngine, MBTI schema, and MBTIScorer at final stage.
Plotting logic unchanged from original.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
from pathlib import Path

import sys
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from synthetic_user import SyntheticUser

from adaptive_quiz.core import AdaptiveEngine, generate_question_weights
from adaptive_quiz.scoring import MBTIScorer


def _load_schema(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def run_simulation(
    d: int = 4,
    T: int = 30,
    seed: int = 42,
    selection_mode: str = "variance",
    schema_path: Path | None = None,
    use_synthetic_pool: bool = False,
):
    """
    Run simulation with AdaptiveEngine. Uses MBTI schema by default, or synthetic pool if use_synthetic_pool=True.

    Returns
    -------
    mu_traj, Sigma_traj, responses, response_times, w_list, user
    """
    rng = np.random.RandomState(seed)
    user = SyntheticUser(d=d, seed=seed)

    if use_synthetic_pool:
        w_list = generate_question_weights(d, T, random_state=rng)
        schema = {
            "name": "Synthetic",
            "dimensions": [{"name": f"dim_{i}"} for i in range(d)],
            "questions": [{"id": f"q{i}", "text": "", "weights": w.tolist()} for i, w in enumerate(w_list)],
        }
    else:
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "schemas" / "mbti.json"
        schema = _load_schema(schema_path)
        w_list = [np.array(q["weights"]) for q in schema["questions"]]
        T = min(T, len(w_list))

    engine = AdaptiveEngine(
        schema=schema,
        stopping_rule=None,
        selection_mode=selection_mode,
    )

    responses = []
    response_times = []
    w_list_ordered = []
    mu_traj = [engine.get_state()["mu"].copy()]
    Sigma_traj = [engine.get_state()["Sigma"].copy()]

    for _ in range(T):
        next_q = engine.get_next_question()
        if next_q is None:
            break
        question_id, w_t = next_q
        y_t, r_t, _ = user.simulate_response(w_t)
        responses.append(y_t)
        response_times.append(r_t)
        w_list_ordered.append(w_t.copy())
        engine.submit_answer(question_id, y_t, r_t)
        state = engine.get_state()
        mu_traj.append(state["mu"].copy())
        Sigma_traj.append(state["Sigma"].copy())

    mu_traj = np.array(mu_traj)
    Sigma_traj = np.array(Sigma_traj)
    return mu_traj, Sigma_traj, responses, response_times, w_list_ordered, user


def compute_metrics(mu_traj, Sigma_traj, true_theta=None):
    """Uncertainty trace, error from true theta (if provided), step magnitudes."""
    uncertainty_trace = np.array([np.trace(Sigma_traj[i]) for i in range(len(Sigma_traj))])
    step_magnitudes = np.array(
        [0.0] + [np.linalg.norm(mu_traj[i] - mu_traj[i - 1]) for i in range(1, len(mu_traj))]
    )
    error_from_true = None
    if true_theta is not None:
        error_from_true = np.array([np.linalg.norm(mu_traj[i] - true_theta) for i in range(len(mu_traj))])
    return uncertainty_trace, error_from_true, step_magnitudes


def animate_simulation_2d(mu_traj, Sigma_traj, T, colors, true_theta=None, show_metrics=True, show_step_size=True):
    """Animate 2D projection with metrics overlay."""
    uncertainty_trace, error_from_true, step_magnitudes = compute_metrics(mu_traj, Sigma_traj, true_theta)
    fig = plt.figure(figsize=(10, 6))
    ax_main = plt.subplot(1, 2, 1)
    ax_metrics = plt.subplot(1, 2, 2)
    scat = ax_main.scatter([], [], s=30)
    ellipse_patch = None
    path_points = []
    line_unc, = ax_metrics.plot([], [], "b-", label="Uncertainty (trace)")
    line_err = None
    if error_from_true is not None:
        line_err, = ax_metrics.plot([], [], "r-", label="Error from true")
    ax_main.set_xlim(-1.5, 1.5)
    ax_main.set_ylim(-1.5, 1.5)
    ax_main.set_xlabel("Trait 1")
    ax_main.set_ylabel("Trait 2")
    ax_main.set_title("Adaptive Bayesian Latent Trait Simulation")
    ax_main.grid(True)
    if show_metrics:
        ax_metrics.set_xlabel("Step")
        ax_metrics.set_ylabel("Metric Value")
        ax_metrics.set_title("Real-time Metrics")
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
        ax_metrics.set_xlim(0, T)
        if error_from_true is not None:
            ax_metrics.set_ylim(0, max(np.max(uncertainty_trace), np.max(error_from_true)) * 1.1)
        else:
            ax_metrics.set_ylim(0, np.max(uncertainty_trace) * 1.1)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return [scat]

    def update(frame):
        nonlocal ellipse_patch
        t = frame + 1
        path_points.append(mu_traj[t, :2])
        offsets = np.array(path_points)
        alphas = np.exp(-0.2 * (len(path_points) - 1 - np.arange(len(path_points))))
        if show_step_size and t > 0:
            sizes = [30 + 200 * step_magnitudes[min(i + 1, len(step_magnitudes) - 1)] for i in range(len(path_points))]
        else:
            sizes = [30] * len(path_points)
        rgba_colors = [to_rgba(colors[i], alpha=alphas[i]) for i in range(len(path_points))]
        scat.set_offsets(offsets)
        scat.set_facecolor(rgba_colors)
        scat.set_sizes(sizes)
        if ellipse_patch is not None:
            ellipse_patch.remove()
        Sigma_2d = Sigma_traj[t][:2, :2]
        eigvals, eigvecs = np.linalg.eigh(Sigma_2d)
        angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))
        width, height = 2 * np.sqrt(eigvals)
        ellipse_patch = Ellipse(
            xy=mu_traj[t, :2], width=width, height=height, angle=angle,
            edgecolor=colors[t - 1], fc="none", lw=1, alpha=0.5,
        )
        ax_main.add_patch(ellipse_patch)
        if show_metrics:
            line_unc.set_data(range(t + 1), uncertainty_trace[: t + 1])
            if line_err is not None:
                line_err.set_data(range(t + 1), error_from_true[: t + 1])
        return [scat, ellipse_patch]

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=300, repeat=False)
    plt.tight_layout()
    return ani


def plot_radar(mu_traj, Sigma_traj, T, true_theta=None, step=None):
    """Radar plot for all trait dimensions (final or at step)."""
    if step is None:
        step = len(mu_traj) - 1
    d = mu_traj.shape[1]
    angles = np.linspace(0, 2 * np.pi, d, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    values = mu_traj[step].tolist()
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label="Current Estimate", color="blue")
    ax.fill(angles, values, alpha=0.25, color="blue")
    if true_theta is not None:
        true_values = true_theta.tolist()
        true_values += true_values[:1]
        ax.plot(angles, true_values, "s-", linewidth=2, label="True Traits", color="red", linestyle="--")
    std_values = np.sqrt(np.diag(Sigma_traj[step])).tolist()
    std_values += std_values[:1]
    upper = [values[i] + std_values[i] for i in range(len(values))]
    lower = [values[i] - std_values[i] for i in range(len(values))]
    ax.fill_between(angles, lower, upper, alpha=0.1, color="blue", label="±1σ uncertainty")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"Trait {i+1}" for i in range(d)])
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"Multi-Axis Trait Profile (Step {step})", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    plt.tight_layout()
    return fig


def animate_radar(mu_traj, Sigma_traj, T, colors, true_theta=None):
    """Animate radar plot over time."""
    d = mu_traj.shape[1]
    angles = np.linspace(0, 2 * np.pi, d, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    line_estimate, = ax.plot([], [], "o-", linewidth=2, label="Current Estimate", color="blue")
    fill_estimate = None
    fill_uncertainty = None
    if true_theta is not None:
        true_values = true_theta.tolist()
        true_values += true_values[:1]
        ax.plot(angles, true_values, "s-", linewidth=2, label="True Traits", color="red", linestyle="--", markersize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"Trait {i+1}" for i in range(d)])
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Animated Multi-Axis Trait Profile", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    def init():
        line_estimate.set_data([], [])
        return [line_estimate]

    def update(frame):
        nonlocal fill_estimate, fill_uncertainty
        t = frame + 1
        values = mu_traj[t].tolist()
        values += values[:1]
        line_estimate.set_data(angles, values)
        if fill_estimate is not None:
            fill_estimate.remove()
        fill_estimate = ax.fill(angles, values, alpha=0.25, color=colors[t - 1])[0]
        if fill_uncertainty is not None:
            fill_uncertainty.remove()
        std_values = np.sqrt(np.diag(Sigma_traj[t])).tolist()
        std_values += std_values[:1]
        upper = [values[i] + std_values[i] for i in range(len(values))]
        lower = [values[i] - std_values[i] for i in range(len(values))]
        fill_uncertainty = ax.fill_between(angles, lower, upper, alpha=0.15, color=colors[t - 1], label="±1σ uncertainty" if t == 1 else "")
        return [line_estimate, fill_estimate, fill_uncertainty]

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=300, repeat=False)
    plt.tight_layout()
    return ani


def animate_parallel_coords(mu_traj, Sigma_traj, T, colors, true_theta=None):
    """Animate parallel coordinates."""
    d = mu_traj.shape[1]
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.linspace(0, 1, d)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Trait {i+1}" for i in range(d)])
    ax.set_ylabel("Trait Value")
    ax.set_title("Parallel Coordinates: All Trait Dimensions", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    line_true = None
    if true_theta is not None:
        line_true, = ax.plot(x_positions, true_theta, "r--", linewidth=2, label="True Traits", alpha=0.7, zorder=1)
    trajectory_lines = []
    uncertainty_bands = []

    def init():
        return []

    def update(frame):
        t = frame + 1
        current_line, = ax.plot(x_positions, mu_traj[t], "o-", linewidth=2.5, color=colors[t - 1], alpha=0.8, markersize=8, label="Current Estimate" if t == 1 else "", zorder=3)
        trajectory_lines.append(current_line)
        if len(trajectory_lines) > 1:
            for i, line in enumerate(trajectory_lines[:-1]):
                alpha = 0.3 * np.exp(-0.1 * (len(trajectory_lines) - 1 - i))
                line.set_alpha(alpha)
                line.set_linewidth(1.5)
        std_values = np.sqrt(np.diag(Sigma_traj[t]))
        upper = mu_traj[t] + std_values
        lower = mu_traj[t] - std_values
        if len(uncertainty_bands) > 0:
            uncertainty_bands[-1].remove()
        band = ax.fill_between(x_positions, lower, upper, alpha=0.2, color=colors[t - 1], zorder=2)
        uncertainty_bands.append(band)
        if line_true is not None:
            ax.legend(loc="upper right")
        return trajectory_lines + [uncertainty_bands[-1]] if uncertainty_bands else trajectory_lines

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=300, repeat=False)
    plt.tight_layout()
    return ani


def animate_multi_2d(mu_traj, Sigma_traj, T, colors, true_theta=None):
    """Animate small multiples of 2D projections."""
    d = mu_traj.shape[1]
    n_pairs = d * (d - 1) // 2
    n_cols = int(np.ceil(np.sqrt(n_pairs)))
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    scatters = []
    ellipse_patches = [None] * n_pairs
    path_points_list = [[] for _ in range(n_pairs)]
    pair_idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            ax = axes[pair_idx]
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlabel(f"Trait {i+1}")
            ax.set_ylabel(f"Trait {j+1}")
            ax.set_title(f"Traits {i+1} vs {j+1}")
            ax.grid(True, alpha=0.3)
            scat = ax.scatter([], [], s=30)
            scatters.append(scat)
            pair_idx += 1
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis("off")

    def init():
        for scat in scatters:
            scat.set_offsets(np.empty((0, 2)))
        return scatters

    def update(frame):
        t = frame + 1
        pair_idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                path_points_list[pair_idx].append(mu_traj[t, [i, j]])
                offsets = np.array(path_points_list[pair_idx])
                alphas = np.exp(-0.2 * (len(path_points_list[pair_idx]) - 1 - np.arange(len(path_points_list[pair_idx]))))
                rgba_colors = [to_rgba(colors[k], alpha=alphas[k]) for k in range(len(path_points_list[pair_idx]))]
                scatters[pair_idx].set_offsets(offsets)
                scatters[pair_idx].set_facecolor(rgba_colors)
                if ellipse_patches[pair_idx] is not None:
                    ellipse_patches[pair_idx].remove()
                Sigma_2d = Sigma_traj[t][np.ix_([i, j], [i, j])]
                eigvals, eigvecs = np.linalg.eigh(Sigma_2d)
                angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))
                width, height = 2 * np.sqrt(eigvals)
                ellipse_patches[pair_idx] = Ellipse(
                    xy=mu_traj[t, [i, j]], width=width, height=height, angle=angle,
                    edgecolor=colors[t - 1], fc="none", lw=1, alpha=0.5,
                )
                axes[pair_idx].add_patch(ellipse_patches[pair_idx])
                pair_idx += 1
        return scatters + [p for p in ellipse_patches if p is not None]

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=300, repeat=False)
    plt.tight_layout()
    return ani


def animate_simulation(mu_traj, Sigma_traj, T, colors, mode="2d", true_theta=None, show_metrics=True, show_step_size=True):
    """Dispatch to 2d, radar, radar_static, multi_2d, or parallel."""
    if mode == "2d":
        return animate_simulation_2d(mu_traj, Sigma_traj, T, colors, true_theta, show_metrics, show_step_size)
    if mode == "radar":
        return animate_radar(mu_traj, Sigma_traj, T, colors, true_theta)
    if mode == "radar_static":
        return plot_radar(mu_traj, Sigma_traj, T, true_theta)
    if mode == "multi_2d":
        return animate_multi_2d(mu_traj, Sigma_traj, T, colors, true_theta)
    if mode == "parallel":
        return animate_parallel_coords(mu_traj, Sigma_traj, T, colors, true_theta)
    raise ValueError(f"Unknown mode: {mode}. Use '2d', 'radar', 'radar_static', 'multi_2d', or 'parallel'")


if __name__ == "__main__":
    d = 4
    T = 30
    seed = 42
    selection_mode = "variance"
    viz_mode = "radar"
    use_synthetic_pool = True  # Set False to use MBTI schema

    mu_traj, Sigma_traj, responses, response_times, w_list, user = run_simulation(
        d=d, T=T, seed=seed, selection_mode=selection_mode, use_synthetic_pool=use_synthetic_pool
    )

    if not use_synthetic_pool:
        scorer = MBTIScorer()
        result = scorer.score(mu_traj[-1], Sigma_traj[-1])
        print("MBTI result:", result["type"], "axis_scores:", result["axis_scores"])

    colors = plt.cm.viridis(np.linspace(0, 1, T))
    ani = animate_simulation(
        mu_traj, Sigma_traj, T, colors, mode=viz_mode,
        true_theta=user.theta_true, show_metrics=True, show_step_size=True
    )
    plt.show()
