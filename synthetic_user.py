import numpy as np

"""
CHANGES:
- Added user archetype subclasses: ConfidentUser, HesitantUser, FragmentedUser
- Each archetype modifies response behavior (speed, noise, consistency)
- Base SyntheticUser remains unchanged for backward compatibility
"""

class SyntheticUser:
    """
    Synthetic user with latent traits and response simulation.
    Base class with standard response behavior.
    """
    def __init__(self, d, seed=2):
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


class ConfidentUser(SyntheticUser):
    """
    Confident user: fast responses, low noise, consistent answers.
    Characteristics: quick decisions, high certainty, low variance.
    """
    def simulate_response(self, w_t):
        # Lower base noise, faster responses
        y_cont = w_t @ self.theta_true + self.rng.normal(0, 0.1)  # Lower noise
        y_t = self.likert_values[np.argmin(np.abs(self.likert_values - y_cont))]
        # Faster response time (multiply by 0.5)
        r_t = self.rng.uniform(0.3, 1.0) * (1.5 - abs(y_t)) * 0.5
        # Lower noise variance (confident = less uncertainty)
        sigma2_t = 0.05 + 0.05 * r_t + 0.05 * (1 - abs(y_t))
        return y_t, r_t, sigma2_t


class HesitantUser(SyntheticUser):
    """
    Hesitant user: slow responses, high noise, uncertain answers.
    Characteristics: takes time to decide, high uncertainty, variable responses.
    """
    def simulate_response(self, w_t):
        # Higher base noise, slower responses
        y_cont = w_t @ self.theta_true + self.rng.normal(0, 0.4)  # Higher noise
        y_t = self.likert_values[np.argmin(np.abs(self.likert_values - y_cont))]
        # Slower response time (multiply by 2.0)
        r_t = self.rng.uniform(1.0, 4.0) * (1.5 - abs(y_t)) * 2.0
        # Higher noise variance (hesitant = more uncertainty)
        sigma2_t = 0.2 + 0.2 * r_t + 0.2 * (1 - abs(y_t))
        return y_t, r_t, sigma2_t


class FragmentedUser(SyntheticUser):
    """
    Fragmented user: inconsistent responses along some trait axes.
    Characteristics: may flip responses for certain traits, creating confusion.
    """
    def __init__(self, d, seed=2, flip_probability=0.3, flip_axes=None):
        """
        Parameters:
        - flip_probability: probability of flipping response for each question
        - flip_axes: list of axis indices to flip (None = random subset)
        """
        super().__init__(d, seed)
        self.flip_probability = flip_probability
        if flip_axes is None:
            # Randomly select some axes to flip
            num_flip = max(1, int(d * 0.3))
            self.flip_axes = set(self.rng.choice(d, num_flip, replace=False))
        else:
            self.flip_axes = set(flip_axes)
    
    def simulate_response(self, w_t):
        # Check if this question probes a flipped axis
        probes_flipped = any(w_t[axis] != 0 for axis in self.flip_axes)
        
        if probes_flipped and self.rng.random() < self.flip_probability:
            # Flip the true trait for this question
            theta_flipped = self.theta_true.copy()
            for axis in self.flip_axes:
                if w_t[axis] != 0:
                    theta_flipped[axis] = -theta_flipped[axis]
            y_cont = w_t @ theta_flipped + self.rng.normal(0, 0.25)
        else:
            # Normal response
            y_cont = w_t @ self.theta_true + self.rng.normal(0, 0.2)
        
        y_t = self.likert_values[np.argmin(np.abs(self.likert_values - y_cont))]
        r_t = self.rng.uniform(0.5, 2.0) * (1.5 - abs(y_t))
        # Slightly higher noise due to inconsistency
        sigma2_t = 0.12 + 0.12 * r_t + 0.12 * (1 - abs(y_t))
        return y_t, r_t, sigma2_t
