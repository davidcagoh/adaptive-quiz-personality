import numpy as np

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

def select_next_question(mu, Sigma, question_pool, asked_indices=None):
    """
    Select the next question to maximize posterior variance reduction.
    
    Parameters:
    - mu: current posterior mean (d,)
    - Sigma: current posterior covariance (d,d)
    - question_pool: list or array of weight vectors (num_questions, d)
    - asked_indices: set/list of already asked question indices
    
    Returns:
    - next_index: index of the selected question
    - w_next: weight vector for the selected question
    """
    if asked_indices is None:
        asked_indices = set()
        
    # Compute score = projected variance along each question's axis
    scores = []
    for idx, w in enumerate(question_pool):
        if idx in asked_indices:
            scores.append(-np.inf)  # skip already asked
            continue
        # projected variance = w^T Sigma w
        score = w.T @ Sigma @ w
        scores.append(score)
    
    next_index = int(np.argmax(scores))
    w_next = question_pool[next_index]
    return next_index, w_next