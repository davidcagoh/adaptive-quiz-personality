import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

# Simulation parameters
d = 4  # number of latent traits
T = 30  # number of questions
np.random.seed(42)

# Synthetic user true traits (-1 to 1)
theta_true = np.random.uniform(-1, 1, d)

# Initialize prior
mu = np.zeros(d)
Sigma = np.eye(d)  # initial uncertainty

# Question weights (randomly probe 1 or 2 axes)
w_list = []
for _ in range(T):
    w = np.zeros(d)
    axes = np.random.choice(d, np.random.randint(1,3), replace=False)
    w[axes] = 1.0
    w_list.append(w)

# Simulate responses on 5-point Likert (-1, -0.5, 0, 0.5, 1)
likert_values = np.array([-1, -0.5, 0, 0.5, 1])
responses = []
response_times = []
mu_traj = [mu.copy()]
Sigma_traj = [Sigma.copy()]

for t in range(T):
    w_t = w_list[t]

    # Simulate continuous ideal response + noise
    y_cont = w_t @ theta_true + np.random.normal(0, 0.2)

    # Map to nearest Likert value
    y_t = likert_values[np.argmin(np.abs(likert_values - y_cont))]
    responses.append(y_t)

    # Simulate response time (faster for stronger opinions)
    r_t = np.random.uniform(0.5, 2.0) * (1.5 - abs(y_t))
    response_times.append(r_t)

    # Noise: function of response time + distance from neutral
    sigma2_t = 0.1 + 0.1 * r_t + 0.1*(1 - abs(y_t))

    # Bayesian update
    Sigma = np.linalg.inv(np.linalg.inv(Sigma) + np.outer(w_t, w_t)/sigma2_t)
    mu = Sigma @ (np.linalg.inv(Sigma_traj[-1]) @ mu_traj[-1] + w_t*y_t/sigma2_t)

    # record trajectory
    mu_traj.append(mu.copy())
    Sigma_traj.append(Sigma.copy())

# Convert trajectory to arrays
mu_traj = np.array(mu_traj)
Sigma_traj = np.array(Sigma_traj)

# Animation setup
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.viridis(np.linspace(0,1,T))
scat = ax.scatter([], [], s=30)
ellipse_patch = None
path_points = []

ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Adaptive Bayesian Latent Trait Simulation with Response Time')
plt.grid(True)

# Function to initialize animation
def init():
    scat.set_offsets(np.empty((0, 2)))
    return [scat]

from matplotlib.colors import to_rgba

def update(frame):
    global ellipse_patch
    t = frame + 1
    path_points.append(mu_traj[t,:2])
    offsets = np.array(path_points)

    # Decaying alpha, older points fade (alpha decreases with index)
    alphas = np.exp(-0.2 * (len(path_points) - 1 - np.arange(len(path_points))))

    # Combine color and alpha into RGBA
    rgba_colors = [to_rgba(colors[i], alpha=alphas[i]) for i in range(len(path_points))]

    scat.set_offsets(offsets)
    scat.set_facecolor(rgba_colors)

    # Remove previous ellipse
    if ellipse_patch is not None:
        ellipse_patch.remove()

    # Draw current ellipse
    Sigma_2d = Sigma_traj[t][:2,:2]
    eigvals, eigvecs = np.linalg.eigh(Sigma_2d)
    angle = np.degrees(np.arctan2(*eigvecs[:,1][::-1]))
    width, height = 2*np.sqrt(eigvals)
    ellipse_patch = Ellipse(
        xy=mu_traj[t,:2],
        width=width, height=height,
        angle=angle,
        edgecolor=colors[t-1],
        fc='none',
        lw=1,
        alpha=0.5
    )
    ax.add_patch(ellipse_patch)

    return [scat, ellipse_patch]

ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=300, repeat=False)
plt.show()