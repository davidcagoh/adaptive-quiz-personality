import numpy as np
import matplotlib.pyplot as plt

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
mu_traj = [mu.copy()]
Sigma_traj = [Sigma.copy()]

for t in range(T):
    w_t = w_list[t]
    # simulate ideal continuous response + noise
    y_cont = w_t @ theta_true + np.random.normal(0, 0.2)
    # map to nearest Likert value
    y_t = likert_values[np.argmin(np.abs(likert_values - y_cont))]
    responses.append(y_t)

    # Simple noise model: higher magnitude = lower variance (simulate confidence)
    sigma2_t = 0.1 + 0.1*(1 - abs(y_t))

    # Bayesian update
    Sigma = np.linalg.inv(np.linalg.inv(Sigma) + np.outer(w_t, w_t)/sigma2_t)
    mu = Sigma @ (np.linalg.inv(Sigma_traj[-1]) @ mu_traj[-1] + w_t*y_t/sigma2_t)

    # record trajectory
    mu_traj.append(mu.copy())
    Sigma_traj.append(Sigma.copy())

# Convert trajectory to arrays
mu_traj = np.array(mu_traj)
Sigma_traj = np.array(Sigma_traj)

# Plot trajectory on first two traits with uncertainty ellipses
fig, ax = plt.subplots(figsize=(6,6))
colors = plt.cm.viridis(np.linspace(0,1,T))

for t in range(1, T+1):
    # Plot point
    ax.scatter(mu_traj[t,0], mu_traj[t,1], color=colors[t-1], s=30)

    # Plot ellipse (simple 2D projection)
    Sigma_2d = Sigma_traj[t][:2,:2]
    eigvals, eigvecs = np.linalg.eigh(Sigma_2d)
    angle = np.degrees(np.arctan2(*eigvecs[:,1][::-1]))
    width, height = 2*np.sqrt(eigvals)  # 1-sigma ellipse
    from matplotlib.patches import Ellipse
    ell = Ellipse(xy=mu_traj[t,:2], width=width, height=height, angle=angle, edgecolor=colors[t-1], fc='none', lw=1, alpha=0.5)
    ax.add_patch(ell)

ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Adaptive Bayesian Latent Trait Simulation')
plt.grid(True)
plt.show()