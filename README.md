# Adaptive Bayesian MBTI Quiz

An adaptive personality quiz engine that uses Bayesian inference to converge on a Myers-Briggs type in roughly half the questions a standard fixed-length test requires.

**[Try it live →](https://adaptive-quiz-personality.vercel.app)**

---

## Results (1,000-user simulation)

| Method | Avg. questions | Type accuracy |
|--------|---------------|--------------|
| Full test (80 questions) | 80 | ~91% |
| **This engine (adaptive)** | **39.8** | **89.7%** |
| Random order + early stopping | 43.9 | 91.4% |
| 16personalities (reference) | ~93 | — |

50% fewer questions than an exhaustive test. Per-axis accuracy above 96% on all four dimensions (EI 97.4%, NS 97.4%, TF 96.6%, JP 98.2%).

![Convergence comparison](assets/convergence.png)

See [REPORT.md](REPORT.md) for the full technical writeup — observation model, Bayesian update derivation, selection strategy, schema design rationale, and experiment results.

---

## How it works

Each question has a 4D weight vector across MBTI axes. After each response, a Gaussian conjugate update narrows the posterior over latent trait space:

```
y_t = w_t^T θ + ε_t,   ε_t ~ N(0, σ²_t)

Σ_post = (Σ_prior⁻¹ + w_t w_t^T / σ²_t)⁻¹
μ_post = Σ_post (Σ_prior⁻¹ μ_prior + w_t y_t / σ²_t)
```

The next question is chosen to maximally reduce remaining uncertainty (argmax **w**ᵀ**Σw**). The quiz stops once posterior variance falls below a threshold on all four axes.

---

## Running locally

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Start quiz (backend + frontend at http://127.0.0.1:8000)
python -m uvicorn backend.main:app --reload
```

**Simulations and experiments:**
```bash
python -m adaptive_quiz.simulation.simulate_convergence   # 3-way adaptive vs random vs full
python -m adaptive_quiz.experiments.threshold_sweep       # accuracy/efficiency Pareto curve
python -m adaptive_quiz.simulation.run_simulation
python -m adaptive_quiz.experiments.run_experiments
```

**Tests:**
```bash
python -m pytest test_modules.py
```

---

## Architecture

```
adaptive_quiz/core/          Domain-agnostic Bayesian engine
adaptive_quiz/domains/mbti/  MBTI schema + scoring
adaptive_quiz/simulation/    Synthetic users for benchmarking
adaptive_quiz/experiments/   Multi-strategy comparison framework
backend/                     FastAPI REST API
frontend/index.html          Single-page quiz UI
```

The engine is domain-agnostic — adding a new personality framework requires only a schema JSON and a scoring function. See [REPORT.md](REPORT.md) for extension instructions and design decisions.

---

## Question bank

80-item IPIP-NEO subset (Goldberg 1999), public domain. Two facets per MBTI axis chosen for high internal consistency (α > 0.75) and minimal cross-axis bleed. Full rationale in [REPORT.md § Schema Design](REPORT.md#schema-design).

---

## License

MIT.
