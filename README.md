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
  - **Variance heuristic** (default): Chooses question to maximize projected variance reduction
  - **Expected information gain**: Advanced mode using mutual information for optimal selection
  - Avoids repeating previously asked questions
  - Backward compatible: variance mode is default

- **Visualization**
  - **2D mode**: Dynamic scatter point with uncertainty ellipse and decaying path
  - **Radar plot**: Multi-axis trait profile with uncertainty bands
  - **Multi-2D**: Small multiples showing all trait pairs simultaneously
  - Real-time metrics overlay: uncertainty trace, error from true traits
  - Step magnitude encoded in point size
  - Color intensity indicates recency / confidence

- **Synthetic Users**
  - **Base SyntheticUser**: Standard response behavior
  - **ConfidentUser**: Fast responses, low noise, consistent answers
  - **HesitantUser**: Slow responses, high noise, uncertain answers
  - **FragmentedUser**: Inconsistent responses along some trait axes

- **Experimental Analysis**
  - Compare adaptive vs random question selection
  - Per-axis posterior variance and error tracking
  - Statistical validation: t-tests, effect sizes, convergence metrics
  - Export results to CSV/JSON for reproducibility
  - Comprehensive plots: trajectories, per-axis metrics, boxplots

---

## Requirements

- Python 3.8+  
- Packages:
  ```bash
  pip install numpy matplotlib scipy
  ```

---

## Project Structure

```
adaptive-quiz-personality/
│
├─ bayesian_update.py            # Bayesian update logic
├─ adaptive_question_selector.py # Question generation and adaptive selection
├─ synthetic_user.py             # Simulated user responses (with archetypes)
├─ run_simulation.py             # Orchestrator: simulation + animation
├─ run_experiments.py            # Experimental comparison with statistics
├─ test_modules.py               # Unit tests for modularity verification
├─ archive/                      # Archived prototype versions
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
- Displays a dynamic 2D animation with metrics overlay
- Options: `mode='2d'`, `'radar'`, or `'multi_2d'` for different visualizations

4. **Run experiments**

```bash
python run_experiments.py
```

- Compares adaptive vs random question selection across multiple users
- Generates statistical analysis and saves results to `results/` directory
- Creates comprehensive plots showing convergence and per-axis metrics

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

## Recent Updates

### ✅ Implemented Features

- **Expected Information Gain**: Advanced question selection using mutual information (`mode='info_gain'`)
- **User Archetypes**: `ConfidentUser`, `HesitantUser`, `FragmentedUser` subclasses
- **Multi-Axis Visualization**: Radar plots and small multiples for all trait pairs
- **Metrics Overlay**: Real-time uncertainty trace and error from true traits
- **Experimental Framework**: Statistical validation with t-tests, effect sizes, CSV/JSON export
- **Per-Axis Tracking**: Individual trait variance and error trajectories
- **Unit Tests**: Comprehensive test suite verifying modularity and correctness

### 🔄 Extending the System

- **Question Pool Management**: Load questions from files/database instead of random generation
- **Longitudinal Tracking**: Multi-session or temporal tracking across quiz attempts
- **Exploration/Exploitation**: Add trade-off parameters for question selection
- **Interface Integration**: Feed trajectory and ellipses into GUI or web interface for real-time interaction

---

## References / Inspiration

- Bayesian adaptive testing: Kalman filter analogy for latent trait updates
- MBTI / personality quizzes: Likert-scale responses
- Adam optimizer: analogy for adaptive step sizing using variance / confidence

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

