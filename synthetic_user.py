import numpy as np

class SyntheticUser:
    """
    Synthetic user with latent traits and response simulation.
    """
    def __init__(self, d, seed=42):
        self.d = d
        self.rng = np.random.RandomState(seed)
        # True latent traits in [-1,1]
        self.theta_true = self.rng.uniform(-1, 1, d)
        # Likert scale values
        self.likert_values = np.array([-1, -0.5, 0, 0.5, 1])

    def simulate_response(self, w_t):
        """
        Simulate a response to a question with weight vector w_t.

        Returns:
        - y_t: observed Likert response
        - r_t: response time
        - sigma2_t: noise variance for Bayesian update
        """
        # Continuous ideal response + noise
        y_cont = w_t @ self.theta_true + self.rng.normal(0, 0.2)
        # Map to nearest Likert value
        y_t = self.likert_values[np.argmin(np.abs(self.likert_values - y_cont))]
        # Simulate response time (faster for stronger opinions)
        r_t = self.rng.uniform(0.5, 2.0) * (1.5 - abs(y_t))
        # Noise variance depends on response time and distance from neutral
        sigma2_t = 0.1 + 0.1 * r_t + 0.1 * (1 - abs(y_t))
        return y_t, r_t, sigma2_t
