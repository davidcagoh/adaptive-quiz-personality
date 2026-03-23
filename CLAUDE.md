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

---

## Decision Log

### 2026-03-22 — Schema replacement: 30-item ad-hoc → 80-item IPIP-NEO subset

**Context:** The original `mbti.json` was hand-authored with no validated psychometric source. Several questions carried empirically unfounded cross-loadings (e.g. a social-gathering question loading on both EI and NS), and the item content was not grounded in published reliability data.

**Options considered:**

| Option | Items | Decision |
|--------|-------|----------|
| Keep existing schema | 30 | Rejected — no validation, incorrect cross-loadings |
| IPIP Big Five Factor Markers (Goldberg 1992) | 50 | Rejected — see below |
| IPIP-NEO 8-facet subset (Goldberg 1999) | 80 | **Selected** |
| IPIP-NEO full instrument | 300 | Rejected — see below |

**Why not the 50-item file:**
Contains 10 Emotional Stability (Neuroticism) items with no MBTI counterpart. An adaptive selection engine will pull those items whenever they project high variance onto the posterior, consuming question budget with no MBTI signal. Would require explicit exclusion logic in the selection strategy. The 8-facet subset avoids this by simply not including them in the bank.

**Why not the full 300-item file:**
60 Neuroticism items, O6 Liberalism items with political content ("Tend to vote for liberal/conservative candidates"), and N5 Immoderation items with eating-disorder-adjacent content. Weaker facets dilute adaptive selection quality — the engine wastes turns on low-alpha items when high-alpha items exist in the same dimension.

**The 8 facets selected and why:**

| MBTI | Facets | Rationale |
|------|--------|-----------|
| EI | E1 Friendliness (α=.87), E2 Gregariousness (α=.79) | Purest social-preference facets; E4 Activity Level and E5 Excitement-Seeking conflate physical energy and novelty-seeking with social orientation |
| NS | O1 Imagination (α=.83), O5 Intellect (α=.86) | Abstract/concrete cognitive style; avoids O6 political contamination and O3 Emotionality which bleeds into TF |
| TF | A3 Altruism (α=.77), A6 Sympathy (α=.75) | Warmth/empathy content closest to decision-making style; other A facets (Trust, Morality, Modesty) measure interpersonal orientation rather than T/F |
| JP | C2 Orderliness (α=.82), C5 Self-Discipline (α=.85) | Strongest JP proxies; C3 Dutifulness and C4 Achievement-Striving load on work ethic rather than structural preference |

**Critical weight convention — Agreeableness inversion:**
High Agreeableness = Feeling. This is non-intuitive and is the most common implementation error. In the engine's weight convention (`TF` positive pole = T):
- `+keyed` A items (agreeing = more agreeable = Feeling) → `weight[TF] = -1.0`
- `-keyed` A items (agreeing = less agreeable = Thinking) → `weight[TF] = +1.0`

The `agree_indicates_mbti` field in the source JSON encodes this correctly. The conversion script trusts that field directly rather than computing from `keying`.

**Source files:** `adaptive_quiz/schemas/files_extracted/` — the two IPIP JSON item banks and the explainer document used to make these decisions. Public domain: https://ipip.ori.org

---

## Next Steps

### High priority

- **IRT calibration** — The current weight vectors are uniform (±1.0). Items differ in discrimination power (how strongly they separate trait levels). Fit a 2-parameter logistic (2PL) IRT model to the Open Psychometrics IPIP-NEO dataset (~1M responses, https://openpsychometrics.org/_rawdata/) and replace uniform weights with empirically derived *a*-parameters. This is the single highest-leverage improvement to inference accuracy.

- **Response scale alignment** — The IPIP uses a 1–5 Likert scale; the engine currently maps responses to `{-1, -0.5, 0, 0.5, 1}`. Verify the frontend presents the IPIP standard labels: *Very Inaccurate / Moderately Inaccurate / Neither / Moderately Accurate / Very Accurate* rather than agree/disagree framing.

- **Item ordering** — The current schema groups all items by facet (20 EI items, then 20 NS items, etc.). Administer interleaved across dimensions to reduce acquiescence bias and prevent respondents from pattern-matching the construct being measured. Shuffle at schema load time or randomise per session.

### Medium priority

- **Neuroticism as bonus output** — The 300-item file has 60 Neuroticism items excluded from the adaptive bank. After the quiz converges on the 4 MBTI dimensions, Neuroticism could be scored as a supplementary output using the N1 Anxiety + N3 Depression facets (highest alphas: .83, .88). Would require a 5th dimension in the posterior or a separate post-hoc scoring step.

- **Session persistence** — Currently all sessions are lost on server restart. Add SQLite or Redis-backed session storage. The `session_store.py` interface is already abstracted; swap the implementation.

- **Response time in API** — `backend/main.py` hardcodes `response_time=1.0`. The frontend should capture wall-clock time between question display and answer submission and pass it in the `/answer` payload. The noise model already accepts response time; this is purely a frontend + API wiring task.

- **Error handling on invalid session IDs** — `session_store.py` raises `KeyError` on unknown IDs. Wrap with a 404 response in `backend/main.py`.

### Lower priority

- **Big Five output mode** — The engine already infers a 4D posterior. Outputting raw Big Five scores (E, O, A, C without the MBTI mapping layer) is a one-line change in `scoring.py` and would make results more scientifically interpretable.

- **IPIP-NEO-120 as alternative bank** — Johnson (2014) selected the 4 highest-discriminating items per facet from the 300-item pool. The resulting 120-item instrument is more efficient per item. The 8-facet subset from it would be 32 items — a leaner bank if convergence studies show the current 80 items are redundant.

- **Cross-loading investigation** — The current schema has zero cross-loadings (every item loads on exactly one dimension). Some IPIP items genuinely load on multiple Big Five factors. Introducing empirically validated cross-loadings (from published factor analyses) would give the engine richer per-question information but complicates the weight convention.
