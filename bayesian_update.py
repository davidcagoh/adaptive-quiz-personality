import numpy as np

def bayesian_update(mu_prior, Sigma_prior, w_t, y_t, sigma2_t):
    """
    Perform Bayesian update for Gaussian prior and likelihood.

    Parameters:
    - mu_prior: prior mean vector
    - Sigma_prior: prior covariance matrix
    - w_t: question weight vector
    - y_t: observed response
    - sigma2_t: noise variance

    Returns:
    - mu_post: posterior mean vector
    - Sigma_post: posterior covariance matrix
    """
    Sigma_inv_prior = np.linalg.inv(Sigma_prior)
    Sigma_post = np.linalg.inv(Sigma_inv_prior + np.outer(w_t, w_t) / sigma2_t)
    mu_post = Sigma_post @ (Sigma_inv_prior @ mu_prior + w_t * y_t / sigma2_t)
    return mu_post, Sigma_post
