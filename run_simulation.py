"""
CHANGES:
- Added multi-axis visualization: radar plot and small multiples of 2D projections
- Added real-time metrics overlay: uncertainty trace, error from true theta
- Added step magnitude encoding in point size
- Enhanced animation to accept any mu/Sigma trajectory (modular design)
- Support for different visualization modes: '2d', 'radar', 'multi_2d'
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
import math

from synthetic_user import SyntheticUser
from adaptive_question_selector import generate_question_weights, select_next_question 
from bayesian_update import bayesian_update


def run_simulation(d=4, T=30, seed=42, selection_mode='variance'):
    """
    Run simulation with configurable question selection mode.
    
    Parameters:
    - selection_mode: 'variance' (default) or 'info_gain'
    """
    rng = np.random.RandomState(seed)

    # Initialize synthetic user
    user = SyntheticUser(d=d, seed=seed)

    # Initialize prior
    mu = np.zeros(d)
    Sigma = np.eye(d)

    # Generate question weights
    w_list = generate_question_weights(d, T, random_state=rng)

    # Storage for responses and response times
    responses = []
    response_times = []
    mu_traj = [mu.copy()]
    Sigma_traj = [Sigma.copy()]

    # Keep a set of already asked questions
    asked_indices = set()

    for t in range(T):
        next_idx, w_t = select_next_question(mu, Sigma, w_list, asked_indices, mode=selection_mode)
        asked_indices.add(next_idx)

        # Simulate user response
        y_t, r_t, sigma2_t = user.simulate_response(w_t)

        # Store response and time
        responses.append(y_t)
        response_times.append(r_t)

        # Bayesian update
        mu, Sigma = bayesian_update(mu, Sigma, w_t, y_t, sigma2_t)

        # Record trajectory
        mu_traj.append(mu.copy())
        Sigma_traj.append(Sigma.copy())

    # Convert trajectory lists to arrays
    mu_traj = np.array(mu_traj)
    Sigma_traj = np.array(Sigma_traj)

    return mu_traj, Sigma_traj, responses, response_times, w_list, user


def compute_metrics(mu_traj, Sigma_traj, true_theta=None):
    """
    Compute metrics for visualization overlay.
    
    Returns:
    - uncertainty_trace: trace of covariance at each step
    - error_from_true: ||mu - theta_true|| at each step (if true_theta provided)
    - step_magnitudes: ||mu[t] - mu[t-1]|| for each step
    """
    uncertainty_trace = np.array([np.trace(Sigma_traj[i]) for i in range(len(Sigma_traj))])
    step_magnitudes = np.array([0.0] + [np.linalg.norm(mu_traj[i] - mu_traj[i-1]) 
                                        for i in range(1, len(mu_traj))])
    
    error_from_true = None
    if true_theta is not None:
        error_from_true = np.array([np.linalg.norm(mu_traj[i] - true_theta) 
                                   for i in range(len(mu_traj))])
    
    return uncertainty_trace, error_from_true, step_magnitudes


def animate_simulation_2d(mu_traj, Sigma_traj, T, colors, true_theta=None, 
                          show_metrics=True, show_step_size=True):
    """
    Animate 2D projection with metrics overlay.
    Modular: accepts any mu/Sigma trajectory.
    """
    uncertainty_trace, error_from_true, step_magnitudes = compute_metrics(mu_traj, Sigma_traj, true_theta)
    
    fig = plt.figure(figsize=(10, 6))
    ax_main = plt.subplot(1, 2, 1)
    ax_metrics = plt.subplot(1, 2, 2)
    
    scat = ax_main.scatter([], [], s=30)
    ellipse_patch = None
    path_points = []
    
    # Metrics plots
    line_unc, = ax_metrics.plot([], [], 'b-', label='Uncertainty (trace)')
    line_err = None
    if error_from_true is not None:
        line_err, = ax_metrics.plot([], [], 'r-', label='Error from true')
    
    ax_main.set_xlim(-1.5, 1.5)
    ax_main.set_ylim(-1.5, 1.5)
    ax_main.set_xlabel('Trait 1')
    ax_main.set_ylabel('Trait 2')
    ax_main.set_title('Adaptive Bayesian Latent Trait Simulation')
    ax_main.grid(True)
    
    if show_metrics:
        ax_metrics.set_xlabel('Step')
        ax_metrics.set_ylabel('Metric Value')
        ax_metrics.set_title('Real-time Metrics')
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
        
        # Decaying alpha, older points fade
        alphas = np.exp(-0.2 * (len(path_points) - 1 - np.arange(len(path_points))))
        
        # Step size encoding in point size (if enabled)
        if show_step_size and t > 0:
            sizes = [30 + 200 * step_magnitudes[min(i+1, len(step_magnitudes)-1)] 
                    for i in range(len(path_points))]
        else:
            sizes = [30] * len(path_points)
        
        # Combine color and alpha into RGBA
        rgba_colors = [to_rgba(colors[i], alpha=alphas[i]) for i in range(len(path_points))]
        
        scat.set_offsets(offsets)
        scat.set_facecolor(rgba_colors)
        scat.set_sizes(sizes)
        
        # Remove previous ellipse
        if ellipse_patch is not None:
            ellipse_patch.remove()
        
        # Draw current ellipse
        Sigma_2d = Sigma_traj[t][:2, :2]
        eigvals, eigvecs = np.linalg.eigh(Sigma_2d)
        angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))
        width, height = 2 * np.sqrt(eigvals)
        ellipse_patch = Ellipse(
            xy=mu_traj[t, :2],
            width=width, height=height,
            angle=angle,
            edgecolor=colors[t - 1],
            fc='none',
            lw=1,
            alpha=0.5
        )
        ax_main.add_patch(ellipse_patch)
        
        # Update metrics
        if show_metrics:
            line_unc.set_data(range(t+1), uncertainty_trace[:t+1])
            if line_err is not None:
                line_err.set_data(range(t+1), error_from_true[:t+1])
        
        return [scat, ellipse_patch]
    
    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=300, repeat=False)
    plt.tight_layout()
    return ani


def plot_radar(mu_traj, Sigma_traj, T, true_theta=None, step=None):
    """
    Create radar plot showing all trait dimensions.
    If step is None, shows final state; otherwise shows state at step.
    """
    if step is None:
        step = len(mu_traj) - 1
    
    d = mu_traj.shape[1]
    angles = np.linspace(0, 2 * np.pi, d, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Current estimate
    values = mu_traj[step].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label='Current Estimate', color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    
    # True values (if provided)
    if true_theta is not None:
        true_values = true_theta.tolist()
        true_values += true_values[:1]
        ax.plot(angles, true_values, 's-', linewidth=2, label='True Traits', color='red', linestyle='--')
    
    # Uncertainty bands (1-sigma)
    std_values = np.sqrt(np.diag(Sigma_traj[step])).tolist()
    std_values += std_values[:1]
    upper = [values[i] + std_values[i] for i in range(len(values))]
    lower = [values[i] - std_values[i] for i in range(len(values))]
    ax.fill_between(angles, lower, upper, alpha=0.1, color='blue', label='±1σ uncertainty')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'Trait {i+1}' for i in range(d)])
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f'Multi-Axis Trait Profile (Step {step})', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def animate_multi_2d(mu_traj, Sigma_traj, T, colors, true_theta=None):
    """
    Animate small multiples of 2D projections for all pairs of traits.
    """
    d = mu_traj.shape[1]
    n_pairs = d * (d - 1) // 2
    
    # Create grid of subplots
    n_cols = int(np.ceil(np.sqrt(n_pairs)))
    n_rows = int(np.ceil(n_pairs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    scatters = []
    ellipse_patches = [None] * n_pairs
    path_points_list = [[] for _ in range(n_pairs)]
    
    pair_idx = 0
    for i in range(d):
        for j in range(i+1, d):
            ax = axes[pair_idx]
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlabel(f'Trait {i+1}')
            ax.set_ylabel(f'Trait {j+1}')
            ax.set_title(f'Traits {i+1} vs {j+1}')
            ax.grid(True, alpha=0.3)
            
            scat = ax.scatter([], [], s=30)
            scatters.append(scat)
            pair_idx += 1
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    def init():
        for scat in scatters:
            scat.set_offsets(np.empty((0, 2)))
        return scatters
    
    def update(frame):
        t = frame + 1
        pair_idx = 0
        
        for i in range(d):
            for j in range(i+1, d):
                path_points_list[pair_idx].append(mu_traj[t, [i, j]])
                offsets = np.array(path_points_list[pair_idx])
                
                alphas = np.exp(-0.2 * (len(path_points_list[pair_idx]) - 1 - 
                                       np.arange(len(path_points_list[pair_idx]))))
                rgba_colors = [to_rgba(colors[k], alpha=alphas[k]) 
                              for k in range(len(path_points_list[pair_idx]))]
                
                scatters[pair_idx].set_offsets(offsets)
                scatters[pair_idx].set_facecolor(rgba_colors)
                
                # Update ellipse
                if ellipse_patches[pair_idx] is not None:
                    ellipse_patches[pair_idx].remove()
                
                Sigma_2d = Sigma_traj[t][np.ix_([i, j], [i, j])]
                eigvals, eigvecs = np.linalg.eigh(Sigma_2d)
                angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))
                width, height = 2 * np.sqrt(eigvals)
                ellipse_patches[pair_idx] = Ellipse(
                    xy=mu_traj[t, [i, j]],
                    width=width, height=height,
                    angle=angle,
                    edgecolor=colors[t - 1],
                    fc='none',
                    lw=1,
                    alpha=0.5
                )
                axes[pair_idx].add_patch(ellipse_patches[pair_idx])
                
                pair_idx += 1
        
        return scatters + [p for p in ellipse_patches if p is not None]
    
    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, 
                                  blit=False, interval=300, repeat=False)
    plt.tight_layout()
    return ani


def animate_simulation(mu_traj, Sigma_traj, T, colors, mode='2d', true_theta=None, 
                      show_metrics=True, show_step_size=True):
    """
    Main animation function with multiple visualization modes.
    
    Parameters:
    - mode: '2d' (default), 'radar', or 'multi_2d'
    - true_theta: true trait values for error computation (optional)
    - show_metrics: show real-time metrics overlay (for 2d mode)
    - show_step_size: encode step magnitude in point size (for 2d mode)
    """
    if mode == '2d':
        return animate_simulation_2d(mu_traj, Sigma_traj, T, colors, true_theta, 
                                    show_metrics, show_step_size)
    elif mode == 'radar':
        # For radar, show final state (can be extended to animate)
        return plot_radar(mu_traj, Sigma_traj, T, true_theta)
    elif mode == 'multi_2d':
        return animate_multi_2d(mu_traj, Sigma_traj, T, colors, true_theta)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use '2d', 'radar', or 'multi_2d'")


if __name__ == "__main__":
    d = 4
    T = 30
    seed = 42
    selection_mode = 'variance'  # or 'info_gain'
    viz_mode = '2d'  # '2d', 'radar', or 'multi_2d'
    
    mu_traj, Sigma_traj, responses, response_times, w_list, user = run_simulation(
        d=d, T=T, seed=seed, selection_mode=selection_mode
    )
    
    colors = plt.cm.viridis(np.linspace(0, 1, T))
    
    # Animate with true theta for error computation
    ani = animate_simulation(mu_traj, Sigma_traj, T, colors, mode=viz_mode, 
                            true_theta=user.theta_true, show_metrics=True, show_step_size=True)
    
    # Also show final radar plot
    if viz_mode != 'radar':
        plt.figure()
        plot_radar(mu_traj, Sigma_traj, T, true_theta=user.theta_true)
    
    plt.show()
