import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from matplotlib.colors import to_rgba

from synthetic_user import SyntheticUser
from adaptive_question_selector import generate_question_weights, select_next_question 
from bayesian_update import bayesian_update

def run_simulation(d=4, T=30, seed=1):
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
        next_idx, w_t = select_next_question(mu, Sigma, w_list, asked_indices)
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

def animate_simulation(mu_traj, Sigma_traj, T, colors):
    fig, ax = plt.subplots(figsize=(6,6))
    scat = ax.scatter([], [], s=30)
    ellipse_patch = None
    path_points = []

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('Adaptive Bayesian Latent Trait Simulation with Response Time')
    plt.grid(True)

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

        # Combine color and alpha into RGBA
        rgba_colors = [to_rgba(colors[i], alpha=alphas[i]) for i in range(len(path_points))]

        scat.set_offsets(offsets)
        scat.set_facecolor(rgba_colors)

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
        ax.add_patch(ellipse_patch)

        return [scat, ellipse_patch]

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=300, repeat=False)
    plt.show()

if __name__ == "__main__":
    d = 4
    T = 30
    seed = 42

    mu_traj, Sigma_traj, responses, response_times, w_list, user = run_simulation(d=d, T=T, seed=seed)

    colors = plt.cm.viridis(np.linspace(0, 1, T))
    animate_simulation(mu_traj, Sigma_traj, T, colors)
