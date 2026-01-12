import numpy as np
import matplotlib.pyplot as plt
from synthetic_user import SyntheticUser
from adaptive_question_selector import generate_question_weights, select_next_question
from bayesian_update import bayesian_update

def run_multiple_users_comparison(
    user_generator,
    question_weights_generator,
    question_selector,
    bayesian_updater,
    d=4,
    T=30,
    num_users=20,
    seed=1234
):
    """
    Run multiple synthetic users, comparing adaptive vs random question selection.
    Returns results for plotting: convergence speed and final uncertainty.
    """
    rng = np.random.RandomState(seed)
    results = {
        "adaptive": {"final_uncertainty": [], "uncertainty_traj": [], "mu_error_traj": []},
        "random": {"final_uncertainty": [], "uncertainty_traj": [], "mu_error_traj": []},
    }
    for user_idx in range(num_users):
        user_seed = seed + user_idx
        user = user_generator(d=d, seed=user_seed)
        # Generate question weights (same for both strategies for this user)
        w_list = question_weights_generator(d, T, random_state=np.random.RandomState(user_seed + 10000))
        # True user trait
        true_theta = user.theta_true

        # --- Adaptive selection ---
        mu = np.zeros(d)
        Sigma = np.eye(d)
        asked_indices = set()
        mu_traj = [mu.copy()]
        Sigma_traj = [Sigma.copy()]
        mu_error_traj = [np.linalg.norm(mu - true_theta)]
        for t in range(T):
            next_idx, w_t = question_selector(mu, Sigma, w_list, asked_indices)
            asked_indices.add(next_idx)
            y_t, r_t, sigma2_t = user.simulate_response(w_t)
            mu, Sigma = bayesian_updater(mu, Sigma, w_t, y_t, sigma2_t)
            mu_traj.append(mu.copy())
            Sigma_traj.append(Sigma.copy())
            mu_error_traj.append(np.linalg.norm(mu - true_theta))
        Sigma_traj = np.array(Sigma_traj)
        uncertainty_traj = [np.trace(Sigma_traj[t]) for t in range(len(Sigma_traj))]
        results["adaptive"]["final_uncertainty"].append(uncertainty_traj[-1])
        results["adaptive"]["uncertainty_traj"].append(uncertainty_traj)
        results["adaptive"]["mu_error_traj"].append(mu_error_traj)

        # --- Random selection ---
        mu = np.zeros(d)
        Sigma = np.eye(d)
        asked_indices = set()
        mu_traj = [mu.copy()]
        Sigma_traj = [Sigma.copy()]
        mu_error_traj = [np.linalg.norm(mu - true_theta)]
        rng_user = np.random.RandomState(user_seed + 20000)
        for t in range(T):
            # Pick random unasked question
            remaining_indices = [idx for idx in range(T) if idx not in asked_indices]
            next_idx = rng_user.choice(remaining_indices)
            w_t = w_list[next_idx]
            asked_indices.add(next_idx)
            y_t, r_t, sigma2_t = user.simulate_response(w_t)
            mu, Sigma = bayesian_updater(mu, Sigma, w_t, y_t, sigma2_t)
            mu_traj.append(mu.copy())
            Sigma_traj.append(Sigma.copy())
            mu_error_traj.append(np.linalg.norm(mu - true_theta))
        Sigma_traj = np.array(Sigma_traj)
        uncertainty_traj = [np.trace(Sigma_traj[t]) for t in range(len(Sigma_traj))]
        results["random"]["final_uncertainty"].append(uncertainty_traj[-1])
        results["random"]["uncertainty_traj"].append(uncertainty_traj)
        results["random"]["mu_error_traj"].append(mu_error_traj)

    # Convert lists to arrays for easier plotting
    for k in results:
        results[k]["final_uncertainty"] = np.array(results[k]["final_uncertainty"])
        results[k]["uncertainty_traj"] = np.array(results[k]["uncertainty_traj"])
        results[k]["mu_error_traj"] = np.array(results[k]["mu_error_traj"])
    return results

if __name__ == "__main__":
    d = 4
    T = 30
    num_users = 20
    seed = 42

    # Run adaptive vs random comparison with modular components
    results = run_multiple_users_comparison(
        user_generator=SyntheticUser,
        question_weights_generator=generate_question_weights,
        question_selector=select_next_question,
        bayesian_updater=bayesian_update,
        d=d,
        T=T,
        num_users=num_users,
        seed=seed
    )

    # Plot average uncertainty trajectories
    plt.figure()
    for method in ["adaptive", "random"]:
        mean_uncertainty = results[method]["uncertainty_traj"].mean(axis=0)
        plt.plot(mean_uncertainty, label=method)
    plt.xlabel("Step")
    plt.ylabel("Trace of Posterior Covariance")
    plt.title("Convergence of Uncertainty: Adaptive vs Random")
    plt.legend()
    plt.show()