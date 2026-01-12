# Adaptive Bayesian Personality Quiz Simulation

## Overview

This project simulates an **adaptive personality quiz** using a Bayesian framework. The system estimates latent personality traits over multiple dimensions (e.g., MBTI axes) by:

- Asking a series of questions
- Capturing user responses on a 5-point Likert scale
- Using response time as a measure of confidence
- Updating the latent trait estimates with **Bayesian updates**
- Selecting subsequent questions adaptively to reduce uncertainty efficiently
- Visualizing the latent state evolution with dynamic ellipses and a decaying trajectory

This serves as a **proof-of-concept for adaptive testing**, showing how an “Adam-like” adaptive optimizer analogy can be applied to human trait inference.

---

## Features

- **Modular Design**
  - `synthetic_user.py` → Simulates user responses with noise and response time
  - `bayesian_update.py` → Performs Gaussian Bayesian updates
  - `adaptive_question_selector.py` → Generates question weights and selects next question based on posterior uncertainty
  - `run_simulation.py` → Orchestrates the simulation and renders the animation

- **Adaptive Question Selection**
  - Chooses the next question to maximize projected variance reduction
  - Avoids repeating previously asked questions

- **Visualization**
  - Dynamic scatter point representing current latent estimate
  - Uncertainty ellipse showing covariance
  - Decaying path to illustrate the trajectory of convergence
  - Color intensity indicates recency / confidence

- **Synthetic Users**
  - Supports multiple simulated profiles
  - Can test fast, slow, confident, hesitant, or fragmented responding behavior

---

## Requirements

- Python 3.8+  
- Packages:
  ```bash
  pip install numpy matplotlib
  ```

---

## Project Structure

```
adaptive-quiz-personality/
│
├─ bayesian_update.py            # Bayesian update logic
├─ adaptive_question_selector.py # Question generation and adaptive selection
├─ synthetic_user.py             # Simulated user responses
├─ run_simulation.py             # Orchestrator: simulation + animation
└─ README.md                     # This file
```

---

## Getting Started

1. **Clone the repository**

```bash
git clone <repo_url>
cd adaptive-quiz-personality
```

2. **Install dependencies**

```bash
pip install numpy matplotlib
```

3. **Run the simulation**

```bash
python run_simulation.py
```

- This will simulate a synthetic user taking the adaptive quiz
- Displays a dynamic 2D animation of the first two trait dimensions, including:
  - Latent trait point movement
  - Uncertainty ellipse
  - Decaying path of past states

---

## How it Works

1. **Initialization**
   - Synthetic user is generated with latent traits (`theta_true`)
   - Prior mean (`mu`) is set to zeros
   - Covariance (`Sigma`) is initialized as identity

2. **Question Loop**
   - Select next question adaptively using `select_next_question()`
   - Simulate user response:
     - Likert scale (-1, -0.5, 0, 0.5, 1)
     - Response time scales noise variance
   - Update posterior:
     ```python
     mu, Sigma = bayesian_update(mu, Sigma, w_t, y_t, sigma2_t)
     ```
   - Store trajectory for visualization

3. **Visualization**
   - Latent point moves in 2D projection
   - Ellipse reflects current covariance
   - Path fades over time (decaying trajectory)
   - Color intensity corresponds to recency

---

## Extending the System

- **Add More Traits**
  - Increase `d` in `run_simulation.py`
  - Update visualization to handle multiple axes (radar or multiple 2D projections)

- **Advanced Adaptive Selection**
  - Replace variance heuristic with expected information gain
  - Implement exploration/exploitation trade-offs

- **Multiple User Archetypes**
  - Modify `SyntheticUser` to simulate confident, hesitant, or fragmented response patterns
  - Compare adaptive vs random question ordering

- **Interface Integration**
  - The current system outputs trajectory and ellipses; this can be fed into a GUI or web interface for real-time user interaction

---

## References / Inspiration

- Bayesian adaptive testing: Kalman filter analogy for latent trait updates
- MBTI / personality quizzes: Likert-scale responses
- Adam optimizer: analogy for adaptive step sizing using variance / confidence

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

