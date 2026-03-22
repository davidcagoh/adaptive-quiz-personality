# CLAUDE.md — AI Assistant Guide for `adaptive-quiz-personality`

## Overview

This repository implements an **adaptive Bayesian personality quiz engine** with an MBTI domain adapter. The core innovation is a domain-agnostic inference engine that selects questions to maximally reduce uncertainty about a user's latent trait vector.

---

## Repository Structure

```
adaptive-quiz-personality/
├── adaptive_quiz/                 # Main package (core logic)
│   ├── core/                      # Domain-agnostic Bayesian inference engine
│   │   ├── bayesian.py            # Gaussian Bayesian update (conjugate prior)
│   │   ├── engine.py              # AdaptiveEngine orchestrator
│   │   ├── selection.py           # Question selection strategies (Protocol-based)
│   │   └── stopping.py            # Stopping rules (Protocol-based)
│   ├── domains/mbti/              # MBTI-specific domain adapter
│   │   ├── schema.py              # Loads 30-question MBTI schema from JSON
│   │   └── scoring.py             # Maps posterior → MBTI type string
│   ├── scoring/
│   │   └── mbti_scoring.py        # MBTIScorer class (type, letters, confidence)
│   ├── simulation/
│   │   └── run_simulation.py      # Simulation runner + visualization
│   ├── experiments/
│   │   └── run_experiments.py     # Multi-user experiment comparison framework
│   └── schemas/
│       └── mbti.json              # MBTI question pool (30 questions, 4D weights)
├── backend/                       # FastAPI REST API
│   ├── main.py                    # Endpoints: /start_quiz, /answer, /result
│   ├── models.py                  # Pydantic request/response models
│   ├── schemas.py                 # Schema loader
│   └── session_store.py           # In-memory session storage (dict-based)
├── archive/                       # Legacy prototype scripts (do not modify)
├── results/                       # Experiment output (CSV, JSON, PNG)
├── run_simulation.py              # Root-level entry point (delegates to package)
├── run_experiments.py             # Root-level entry point (delegates to package)
├── run_mbti_demo.py               # MBTI-specific interactive demo
├── synthetic_user.py              # SyntheticUser base + archetypes
├── test_modules.py                # Unit tests (unittest-based)
└── simulate_mbti_convergence.py   # Convergence analysis script
```

---

## Technology Stack

- **Language:** Python 3.8+
- **Core deps:** `numpy`, `scipy`, `matplotlib`
- **API:** `fastapi`, `pydantic`, `uvicorn`
- **Tests:** `unittest` (stdlib); `pytest` compatible
- **No database** — sessions are in-memory only
- **No requirements.txt** — install deps manually

---

## Running the Project

### Run a Simulation
```bash
python run_simulation.py
```

### Run Experiments (Compare Selection Strategies)
```bash
python run_experiments.py
# Outputs: results/experiment_summary.json, results/experiment_plots.png
```

### Run the MBTI Demo
```bash
python run_mbti_demo.py
```

### Start the FastAPI Backend
```bash
python -m uvicorn backend.main:app --reload
# Runs on http://127.0.0.1:8000
```

### Run Tests
```bash
python -m pytest test_modules.py
# or
python test_modules.py
```

---

## Core Architecture

### Observation Model
```
y_t = w_t^T θ + ε_t,   ε_t ~ N(0, σ²_t)
```
- `θ` — latent trait vector (unknown, inferred by Bayesian updates)
- `w_t` — weight vector of question `t` (from schema)
- `y_t` — user's observed response (Likert scale: {-1, -0.5, 0, 0.5, 1})
- `σ²_t` — noise variance (from noise model)

### Bayesian Update (`adaptive_quiz/core/bayesian.py`)
Performs a Gaussian conjugate update:
```
μ_post = Σ_post (Σ_prior⁻¹ μ_prior + w σ⁻² y)
Σ_post = (Σ_prior⁻¹ + w wᵀ σ⁻²)⁻¹
```
Pure function — no side effects or global state.

### Question Selection (`adaptive_quiz/core/selection.py`)
Two strategies (Protocol-based):
- **`VarianceSelection`** *(default)*: argmax `wᵀ Σ w` — picks question projecting maximum current uncertainty
- **`InformationGainSelection`**: argmax `0.5 * log(|Σ_prior|/|Σ_post|)` — maximizes expected mutual information

> **Note:** Experiments show `VarianceSelection` outperforms `InformationGainSelection` in practice. Prefer variance unless specifically investigating information gain.

### Stopping Rules (`adaptive_quiz/core/stopping.py`)
Two rules (Protocol-based):
- **`VarianceThresholdStopping`**: stops when all diagonal variances < threshold
- **`SignConfidenceStopping`**: stops when posterior confidence of correct sign ≥ threshold for all dimensions (uses Gaussian CDF)

### `AdaptiveEngine` (`adaptive_quiz/core/engine.py`)
Central orchestrator. Key methods:
- `get_next_question()` → `(question_id, weight_vector)`
- `submit_answer(question_id, response, response_time)` → updates posterior
- `is_complete()` → checks stopping rule
- `get_state()` → returns `EngineState` snapshot (dict-compatible)

---

## MBTI Domain

- **4 dimensions:** EI (Extraversion–Introversion), SN (Sensing–Intuition), TF (Thinking–Feeling), JP (Judging–Perceiving)
- **Schema:** 30 questions in `adaptive_quiz/schemas/mbti.json`, each with a 4D weight vector
- **Scoring:** Sign of `μ` per dimension determines letter; magnitude maps to preference strength
- **Type string:** e.g., `"INTJ"` — positive sign → first letter (E/S/T/J), negative → second (I/N/F/P)

---

## REST API

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/start_quiz` | Start a new session; returns first question |
| `POST` | `/answer` | Submit a response; returns next question or completion flag |
| `GET` | `/result?session_id=<id>` | Retrieve final MBTI type and dimension scores |

### Example Flow
```
POST /start_quiz  { "schema_name": "mbti" }
→ { "session_id": "abc123", "question_id": "q1", "prompt": "..." }

POST /answer  { "session_id": "abc123", "response": 0.5 }
→ { "question_id": "q7", "prompt": "...", "complete": false }

GET /result?session_id=abc123
→ { "type": "INTJ", "dimensions": {"E-I": -0.7, "S-N": -0.4, "T-F": -0.6, "J-P": -0.8} }
```

**CORS:** Configured for `http://localhost:3000` only. Change in `backend/main.py` if needed.

**Sessions:** In-memory only — all data is lost on server restart.

---

## Synthetic Users (`synthetic_user.py`)

Used for testing and simulation:

| Class | Behavior |
|-------|----------|
| `SyntheticUser` | Base: latent traits from `U[-1, 1]`, standard noise |
| `ConfidentUser` | Fast responses, low noise, extreme opinions |
| `HesitantUser` | Slow responses, high noise, uncertain opinions |
| `FragmentedUser` | Inconsistent on some axes (`flip_probability` param) |

Responses quantized to Likert scale: `{-1, -0.5, 0, 0.5, 1}`

---

## Key Conventions

### Adding a New Domain
1. Create `adaptive_quiz/domains/<domain>/schema.py` — load questions with weight vectors
2. Create `adaptive_quiz/domains/<domain>/scoring.py` — map posterior → domain-specific result
3. Register the schema in `backend/schemas.py` if using the REST API
4. The core engine (`AdaptiveEngine`) requires no changes

### Implementing a New Selection Strategy
Implement the `SelectionStrategy` Protocol from `adaptive_quiz/core/selection.py`:
```python
class MyStrategy:
    def select(self, questions, asked_ids, mu, Sigma) -> str:
        ...  # return question_id
```
Pass it to `AdaptiveEngine(selection_strategy=MyStrategy())`.

### Implementing a New Stopping Rule
Implement the `StoppingRule` Protocol from `adaptive_quiz/core/stopping.py`:
```python
class MyRule:
    def should_stop(self, mu, Sigma, n_asked) -> bool:
        ...
```
Pass it to `AdaptiveEngine(stopping_rule=MyRule())`.

### Implementing a New Noise Model
Implement the `NoiseModel` Protocol from `adaptive_quiz/core/engine.py`:
```python
class MyNoise:
    def variance(self, response, response_time) -> float:
        ...
```
Pass it to `AdaptiveEngine(noise_model=MyNoise())`.

---

## Testing

Tests are in `test_modules.py` using `unittest`. Key test classes:

- `TestBayesianUpdate` — verifies uncertainty reduction, pure function behavior
- `TestAdaptiveQuestionSelector` — tests weight generation and selection modes
- `TestSyntheticUser` — tests user response simulation and archetypes

When adding new features, add corresponding tests in `test_modules.py`.

---

## Known Limitations

- **No persistence:** Sessions and results exist only in RAM. Restart clears all state.
- **CORS hardcoded:** `http://localhost:3000` — update `backend/main.py` for other origins.
- **Response time ignored in API:** `backend/main.py` passes `response_time=1.0` hardcoded; the frontend does not capture it.
- **No authentication:** The REST API has no auth layer.
- **No error handling on invalid session IDs:** Will raise `KeyError` in session store.

---

## File Editing Notes

- **`archive/`** — Legacy prototypes, do not modify.
- **Root-level `run_*.py` files** — Thin delegators for backwards compatibility; keep them minimal.
- **`adaptive_quiz/core/`** — Domain-agnostic; never import MBTI-specific code here.
- **`results/`** — Generated output; do not commit large binary files unless intentional.
