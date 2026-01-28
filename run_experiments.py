"""
CHANGES:
- Added per-axis posterior variance and error tracking
- Added statistical validation (t-tests, effect sizes, convergence metrics)
- Added CSV/JSON export for reproducibility
- Enhanced plots: per-axis uncertainty, error trajectories, boxplots
- Support for different question selection modes (variance vs info_gain)
- Modular design: all computation in helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from scipy import stats
from pathlib import Path
from synthetic_user import SyntheticUser
from adaptive_question_selector import generate_question_weights, select_next_question
from bayesian_update import bayesian_update


def run_single_user_simulation(user, w_list, question_selector, bayesian_updater, 
                               d, T, selection_mode='variance', random_seed=None):
    """
    Run a single user simulation with given question selection strategy.
    
    Returns:
    - mu_traj: trajectory of posterior means (T+1, d)
    - Sigma_traj: trajectory of posterior covariances (T+1, d, d)
    - per_axis_variance_traj: per-axis variance trajectories (T+1, d)
    - per_axis_error_traj: per-axis error trajectories (T+1, d)
    - total_error_traj: total error ||mu - theta_true|| (T+1,)
    """
    true_theta = user.theta_true
    mu = np.zeros(d)
    Sigma = np.eye(d)
    asked_indices = set()
    
    mu_traj = [mu.copy()]
    Sigma_traj = [Sigma.copy()]
    per_axis_variance_traj = [np.diag(Sigma).copy()]
    per_axis_error_traj = [np.abs(mu - true_theta).copy()]
    total_error_traj = [np.linalg.norm(mu - true_theta)]
    
    rng = np.random.RandomState(random_seed) if random_seed is not None else None
    
    for t in range(T):
        if selection_mode == 'random':
            # Random selection
            remaining_indices = [idx for idx in range(T) if idx not in asked_indices]
            if rng is None:
                next_idx = np.random.choice(remaining_indices)
            else:
                next_idx = rng.choice(remaining_indices)
            w_t = w_list[next_idx]
        else:
            # Adaptive selection (variance or info_gain)
            next_idx, w_t = question_selector(mu, Sigma, w_list, asked_indices, mode=selection_mode)
        
        asked_indices.add(next_idx)
        y_t, r_t, sigma2_t = user.simulate_response(w_t)
        mu, Sigma = bayesian_updater(mu, Sigma, w_t, y_t, sigma2_t)
        
        mu_traj.append(mu.copy())
        Sigma_traj.append(Sigma.copy())
        per_axis_variance_traj.append(np.diag(Sigma).copy())
        per_axis_error_traj.append(np.abs(mu - true_theta).copy())
        total_error_traj.append(np.linalg.norm(mu - true_theta))
    
    return {
        'mu_traj': np.array(mu_traj),
        'Sigma_traj': np.array(Sigma_traj),
        'per_axis_variance_traj': np.array(per_axis_variance_traj),
        'per_axis_error_traj': np.array(per_axis_error_traj),
        'total_error_traj': np.array(total_error_traj),
        'uncertainty_traj': np.array([np.trace(Sigma_traj[i]) for i in range(len(Sigma_traj))])
    }


def run_multiple_users_comparison(
    user_generator,
    question_weights_generator,
    question_selector,
    bayesian_updater,
    d=4,
    T=30,
    num_users=20,
    seed=1234,
    selection_modes=['variance', 'random']
):
    """
    Run multiple synthetic users, comparing different question selection strategies.
    
    Returns comprehensive results including per-axis metrics and trajectories.
    """
    rng = np.random.RandomState(seed)
    results = {}
    
    for mode in selection_modes:
        results[mode] = {
            "final_uncertainty": [],
            "uncertainty_traj": [],
            "total_error_traj": [],
            "final_total_error": [],
            "per_axis_variance_traj": [],  # (num_users, T+1, d)
            "per_axis_error_traj": [],     # (num_users, T+1, d)
            "final_per_axis_variance": [],  # (num_users, d)
            "final_per_axis_error": []      # (num_users, d)
        }
    
    for user_idx in range(num_users):
        user_seed = seed + user_idx
        user = user_generator(d=d, seed=user_seed)
        # Generate question weights (same for all strategies for this user)
        w_list = question_weights_generator(d, T, random_state=np.random.RandomState(user_seed + 10000))
        
        for mode in selection_modes:
            sim_result = run_single_user_simulation(
                user, w_list, question_selector, bayesian_updater,
                d, T, selection_mode=mode, random_seed=user_seed + 20000 if mode == 'random' else None
            )
            
            results[mode]["final_uncertainty"].append(sim_result['uncertainty_traj'][-1])
            results[mode]["uncertainty_traj"].append(sim_result['uncertainty_traj'])
            results[mode]["total_error_traj"].append(sim_result['total_error_traj'])
            results[mode]["final_total_error"].append(sim_result['total_error_traj'][-1])
            results[mode]["per_axis_variance_traj"].append(sim_result['per_axis_variance_traj'])
            results[mode]["per_axis_error_traj"].append(sim_result['per_axis_error_traj'])
            results[mode]["final_per_axis_variance"].append(sim_result['per_axis_variance_traj'][-1])
            results[mode]["final_per_axis_error"].append(sim_result['per_axis_error_traj'][-1])
    
    # Convert lists to arrays
    for mode in selection_modes:
        for key in results[mode]:
            results[mode][key] = np.array(results[mode][key])
    
    return results


def compute_statistics(results, selection_modes=['variance', 'random']):
    """
    Compute statistical tests comparing selection modes.
    
    Returns dictionary with t-tests, effect sizes, and convergence metrics.
    """
    stats_dict = {}
    
    # Compare each adaptive mode against random
    for mode in selection_modes:
        if mode == 'random':
            continue
        
        # Final uncertainty comparison
        adaptive_unc = results[mode]["final_uncertainty"]
        random_unc = results["random"]["final_uncertainty"]
        t_stat_unc, p_val_unc = stats.ttest_rel(adaptive_unc, random_unc)
        effect_size_unc = (np.mean(adaptive_unc) - np.mean(random_unc)) / np.std(random_unc)
        
        # Final error comparison
        adaptive_err = results[mode]["final_total_error"]
        random_err = results["random"]["final_total_error"]
        t_stat_err, p_val_err = stats.ttest_rel(adaptive_err, random_err)
        effect_size_err = (np.mean(adaptive_err) - np.mean(random_err)) / np.std(random_err)
        
        # Convergence: steps to reach 50% of initial uncertainty
        initial_unc = results[mode]["uncertainty_traj"][:, 0]
        target_unc = 0.5 * initial_unc
        convergence_steps = []
        for i, traj in enumerate(results[mode]["uncertainty_traj"]):
            below_target = np.where(traj <= target_unc[i])[0]
            if len(below_target) > 0:
                convergence_steps.append(below_target[0])
            else:
                convergence_steps.append(len(traj) - 1)
        
        random_conv_steps = []
        for i, traj in enumerate(results["random"]["uncertainty_traj"]):
            below_target = np.where(traj <= target_unc[i])[0]
            if len(below_target) > 0:
                random_conv_steps.append(below_target[0])
            else:
                random_conv_steps.append(len(traj) - 1)
        
        t_stat_conv, p_val_conv = stats.ttest_rel(convergence_steps, random_conv_steps)
        effect_size_conv = (np.mean(convergence_steps) - np.mean(random_conv_steps)) / np.std(random_conv_steps)
        
        stats_dict[mode] = {
            'uncertainty': {
                't_statistic': float(t_stat_unc),
                'p_value': float(p_val_unc),
                'effect_size': float(effect_size_unc),
                'mean_adaptive': float(np.mean(adaptive_unc)),
                'mean_random': float(np.mean(random_unc)),
                'improvement_pct': float((1 - np.mean(adaptive_unc) / np.mean(random_unc)) * 100)
            },
            'error': {
                't_statistic': float(t_stat_err),
                'p_value': float(p_val_err),
                'effect_size': float(effect_size_err),
                'mean_adaptive': float(np.mean(adaptive_err)),
                'mean_random': float(np.mean(random_err)),
                'improvement_pct': float((1 - np.mean(adaptive_err) / np.mean(random_err)) * 100)
            },
            'convergence': {
                't_statistic': float(t_stat_conv),
                'p_value': float(p_val_conv),
                'effect_size': float(effect_size_conv),
                'mean_adaptive': float(np.mean(convergence_steps)),
                'mean_random': float(np.mean(random_conv_steps)),
                'improvement_pct': float((1 - np.mean(convergence_steps) / np.mean(random_conv_steps)) * 100)
            }
        }
    
    return stats_dict


def save_results(results, stats_dict, output_dir='results', prefix='experiment'):
    """
    Save results to CSV and JSON files for reproducibility.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save summary statistics to JSON
    summary = {
        'statistics': stats_dict,
        'summary_metrics': {}
    }
    
    for mode in results.keys():
        summary['summary_metrics'][mode] = {
            'mean_final_uncertainty': float(np.mean(results[mode]["final_uncertainty"])),
            'std_final_uncertainty': float(np.std(results[mode]["final_uncertainty"])),
            'mean_final_error': float(np.mean(results[mode]["final_total_error"])),
            'std_final_error': float(np.std(results[mode]["final_total_error"]))
        }
    
    with open(output_path / f'{prefix}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save per-user results to CSV
    with open(output_path / f'{prefix}_per_user.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'method', 'final_uncertainty', 'final_error'] + 
                       [f'final_variance_axis_{i}' for i in range(results[list(results.keys())[0]]["final_per_axis_variance"].shape[1])] +
                       [f'final_error_axis_{i}' for i in range(results[list(results.keys())[0]]["final_per_axis_error"].shape[1])])
        
        for mode in results.keys():
            for user_idx in range(len(results[mode]["final_uncertainty"])):
                row = [
                    user_idx, mode,
                    results[mode]["final_uncertainty"][user_idx],
                    results[mode]["final_total_error"][user_idx]
                ]
                row.extend(results[mode]["final_per_axis_variance"][user_idx])
                row.extend(results[mode]["final_per_axis_error"][user_idx])
                writer.writerow(row)
    
    print(f"Results saved to {output_path}/")


def plot_results(results, stats_dict, selection_modes=['variance', 'random']):
    """
    Create comprehensive plots: trajectories, per-axis metrics, and boxplots.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Average uncertainty trajectories
    ax1 = plt.subplot(2, 3, 1)
    for mode in selection_modes:
        mean_uncertainty = results[mode]["uncertainty_traj"].mean(axis=0)
        std_uncertainty = results[mode]["uncertainty_traj"].std(axis=0)
        ax1.plot(mean_uncertainty, label=mode)
        ax1.fill_between(range(len(mean_uncertainty)), 
                         mean_uncertainty - std_uncertainty,
                         mean_uncertainty + std_uncertainty, alpha=0.2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Trace of Posterior Covariance")
    ax1.set_title("Uncertainty Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average error trajectories
    ax2 = plt.subplot(2, 3, 2)
    for mode in selection_modes:
        mean_error = results[mode]["total_error_traj"].mean(axis=0)
        std_error = results[mode]["total_error_traj"].std(axis=0)
        ax2.plot(mean_error, label=mode)
        ax2.fill_between(range(len(mean_error)),
                         mean_error - std_error,
                         mean_error + std_error, alpha=0.2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("||mu - theta_true||")
    ax2.set_title("Posterior Error Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-axis uncertainty trajectories
    ax3 = plt.subplot(2, 3, 3)
    d = results[selection_modes[0]]["per_axis_variance_traj"].shape[2]
    for axis in range(d):
        for mode in selection_modes:
            mean_var = results[mode]["per_axis_variance_traj"][:, :, axis].mean(axis=0)
            ax3.plot(mean_var, label=f'{mode} axis {axis}', linestyle='--' if mode == 'random' else '-', alpha=0.7)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Per-Axis Variance")
    ax3.set_title("Per-Axis Uncertainty")
    ax3.legend(ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-axis error trajectories
    ax4 = plt.subplot(2, 3, 4)
    for axis in range(d):
        for mode in selection_modes:
            mean_err = results[mode]["per_axis_error_traj"][:, :, axis].mean(axis=0)
            ax4.plot(mean_err, label=f'{mode} axis {axis}', linestyle='--' if mode == 'random' else '-', alpha=0.7)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Per-Axis Error |mu - theta_true|")
    ax4.set_title("Per-Axis Error")
    ax4.legend(ncol=2, fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Boxplot: Final uncertainty
    ax5 = plt.subplot(2, 3, 5)
    data_unc = [results[mode]["final_uncertainty"] for mode in selection_modes]
    ax5.boxplot(data_unc, labels=selection_modes)
    ax5.set_ylabel("Final Uncertainty")
    ax5.set_title("Final Uncertainty Distribution")
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Boxplot: Final error
    ax6 = plt.subplot(2, 3, 6)
    data_err = [results[mode]["final_total_error"] for mode in selection_modes]
    ax6.boxplot(data_err, labels=selection_modes)
    ax6.set_ylabel("Final Error ||mu - theta_true||")
    ax6.set_title("Final Error Distribution")
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    d = 4
    T = 30
    num_users = 20
    seed = 42
    selection_modes = ['variance', 'random', 'info_gain']  # Can add 'info_gain' here
    
    # Run comparison
    results = run_multiple_users_comparison(
        user_generator=SyntheticUser,
        question_weights_generator=generate_question_weights,
        question_selector=select_next_question,
        bayesian_updater=bayesian_update,
        d=d,
        T=T,
        num_users=num_users,
        seed=seed,
        selection_modes=selection_modes
    )
    
    # Compute statistics
    stats_dict = compute_statistics(results, selection_modes)
    
    # Print statistics
    print("\n=== Statistical Comparison ===")
    for mode in stats_dict.keys():
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
    
    # Save results
    save_results(results, stats_dict, output_dir='results', prefix='experiment')
    
    # Plot results
    fig = plot_results(results, stats_dict, selection_modes)
    plt.savefig('results/experiment_plots.png', dpi=150, bbox_inches='tight')
    plt.show()