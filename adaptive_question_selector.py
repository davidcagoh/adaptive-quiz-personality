import numpy as np

"""
CHANGES:
- Added expected_information_gain() function to compute mutual information
- Enhanced select_next_question() to support both 'variance' and 'info_gain' modes
- Maintained backward compatibility (default mode is 'variance')
"""

def generate_question_weights(d, T, random_state=None):
    """
    Generate a list of question weight vectors probing 1 or 2 latent traits.

    Parameters:
    - d: number of latent traits
    - T: number of questions
    - random_state: np.random.RandomState or None

    Returns:
    - w_list: list of weight vectors (length T)
    """
    if random_state is None:
        random_state = np.random

    w_list = []
    for _ in range(T):
        w = np.zeros(d)
        num_axes = random_state.randint(1, 3)  # 1 or 2
        axes = random_state.choice(d, num_axes, replace=False)
        w[axes] = 1.0
        w_list.append(w)
    return w_list


def expected_information_gain(Sigma_prior, w, sigma2_t):
    """
    Compute expected information gain (mutual information) for a question.
    
    Information gain = 0.5 * log(det(Sigma_prior) / det(Sigma_post))
    where Sigma_post is the posterior covariance after asking the question.
    
    Parameters:
    - Sigma_prior: current posterior covariance (d, d)
    - w: question weight vector (d,)
    - sigma2_t: noise variance for the question
    
    Returns:
    - info_gain: expected information gain (bits/nats)
    """
    # Compute posterior covariance: Sigma_post = inv(Sigma_prior^-1 + w w^T / sigma2_t)
    Sigma_inv_prior = np.linalg.inv(Sigma_prior)
    Sigma_post = np.linalg.inv(Sigma_inv_prior + np.outer(w, w) / sigma2_t)
    
    # Information gain = 0.5 * log(det(prior) / det(posterior))
    det_prior = np.linalg.det(Sigma_prior)
    det_post = np.linalg.det(Sigma_post)
    
    # Avoid numerical issues with very small determinants
    if det_prior <= 0 or det_post <= 0:
        return 0.0
    
    info_gain = 0.5 * np.log(det_prior / det_post)
    return info_gain


def select_next_question(mu, Sigma, question_pool, asked_indices=None, 
                         mode='variance', sigma2_t=0.2):
    """
    Select the next question to maximize posterior variance reduction or information gain.
    
    Parameters:
    - mu: current posterior mean (d,)
    - Sigma: current posterior covariance (d,d)
    - question_pool: list or array of weight vectors (num_questions, d)
    - asked_indices: set/list of already asked question indices
    - mode: 'variance' (default, backward compatible) or 'info_gain'
    - sigma2_t: noise variance for information gain computation (default 0.2)
    
    Returns:
    - next_index: index of the selected question
    - w_next: weight vector for the selected question
    """
    if asked_indices is None:
        asked_indices = set()
        
    scores = []
    for idx, w in enumerate(question_pool):
        if idx in asked_indices:
            scores.append(-np.inf)  # skip already asked
            continue
        
        if mode == 'info_gain':
            # Expected information gain (mutual information)
            score = expected_information_gain(Sigma, w, sigma2_t)
        else:  # mode == 'variance' (default, backward compatible)
            # Projected variance = w^T Sigma w
            score = w.T @ Sigma @ w
        
        scores.append(score)
    
    next_index = int(np.argmax(scores))
    w_next = question_pool[next_index]
    return next_index, w_next