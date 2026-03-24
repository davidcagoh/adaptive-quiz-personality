# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

An **adaptive Bayesian personality quiz engine** with an MBTI domain adapter. The core innovation is a domain-agnostic inference engine that selects questions to maximally reduce uncertainty about a user's latent trait vector.

---

## Commands

A `.venv` is present at repo root — activate it first: `source .venv/bin/activate`

```bash
# Start the FastAPI backend (serves frontend at http://127.0.0.1:8000)
python -m uvicorn backend.main:app --reload

# Run tests
python -m pytest test_modules.py
python -m pytest test_modules.py::TestBayesianUpdate  # single class

# Run a simulation
python -m adaptive_quiz.simulation.run_simulation

# Compare selection strategies (outputs results/experiment_summary.json + plots)
python -m adaptive_quiz.experiments.run_experiments

# Convergence analysis
python -m adaptive_quiz.simulation.simulate_convergence

# Threshold sensitivity sweep (accuracy vs. questions asked)
python -m adaptive_quiz.experiments.threshold_sweep
```

Install deps: `pip install -r requirements.txt` (numpy, scipy, matplotlib, fastapi, pydantic, uvicorn, supabase).

---

## Repository Structure

```
adaptive_quiz/              # Main Python package
├── core/                   # Domain-agnostic Bayesian engine (never import MBTI here)
│   ├── bayesian.py         # Gaussian conjugate prior update (pure function)
│   ├── engine.py           # AdaptiveEngine orchestrator + NoiseModel protocol
│   ├── selection.py        # SelectionStrategy protocol (VarianceSelection, InformationGainSelection)
│   └── stopping.py         # StoppingRule protocol (VarianceThresholdStopping, SignConfidenceStopping)
├── domains/mbti/           # MBTI domain adapter (schema loader + posterior→type scorer)
├── schemas/
│   ├── mbti.json           # Active question bank (80-item IPIP-NEO subset)
│   └── files_extracted/    # Source IPIP-NEO JSON item banks (reference only)
├── simulation/
│   ├── synthetic_user.py        # SyntheticUser base + ConfidentUser, HesitantUser, FragmentedUser
│   ├── synthetic_schema.py      # generate_synthetic_schema() for complexity experiments
│   ├── run_simulation.py        # Simulation runner + visualization
│   └── simulate_convergence.py  # Adaptive vs full-test convergence comparison
└── experiments/
    ├── run_experiments.py       # Multi-user experiment comparison framework
    └── threshold_sweep.py       # SignConfidenceStopping threshold Pareto analysis
backend/
├── main.py                 # FastAPI app: /start_quiz, /answer, /result; serves frontend/
├── models.py               # Pydantic request/response models
├── schemas.py              # Pluggable schema loader
└── session_store.py        # In-memory session storage (interface is abstracted)
frontend/
└── index.html              # Standalone vanilla JS quiz app (the working UI)
archive/                    # Legacy prototypes — do not modify
results/                    # Experiment output (CSV, JSON, PNG) — do not commit
```

**`run_mbti_demo.py`** is the only root-level script — an interactive CLI demo. All other scripts live in the package and are run with `python -m`.

---

## Core Architecture

### Observation model
```
y_t = w_t^T θ + ε_t,   ε_t ~ N(0, σ²_t)
```
- `θ` — latent trait vector (inferred by Bayesian updates)
- `w_t` — weight vector of question `t` (from schema)
- `y_t` — response mapped to `{-1, -0.5, 0, 0.5, 1}`

### Bayesian update (`adaptive_quiz/core/bayesian.py`)
```
μ_post = Σ_post (Σ_prior⁻¹ μ_prior + w σ⁻² y)
Σ_post = (Σ_prior⁻¹ + w wᵀ σ⁻²)⁻¹
```
Pure function, no side effects.

### Question selection (`adaptive_quiz/core/selection.py`)
- **`VarianceSelection`** *(default)*: argmax `wᵀ Σ w` — projects maximum current uncertainty
- **`InformationGainSelection`**: argmax `0.5 * log(|Σ_prior|/|Σ_post|)`

> Experiments show `VarianceSelection` outperforms `InformationGainSelection` in practice.

### Stopping rules (`adaptive_quiz/core/stopping.py`)
- **`VarianceThresholdStopping`**: all diagonal variances < threshold
- **`SignConfidenceStopping`**: posterior confidence of correct sign ≥ threshold (Gaussian CDF)

### `AdaptiveEngine` (`adaptive_quiz/core/engine.py`)
- `get_next_question()` → `(question_id, weight_vector)`
- `submit_answer(question_id, response, response_time)` → updates posterior
- `is_complete()` / `get_state()` → `EngineState` snapshot

---

## MBTI Domain

- **4 dimensions:** EI, NS, TF, JP
- **Sign convention:** positive `μ` → first letter (E/N/T/J); negative → second (I/S/F/P)
- **Scoring:** `adaptive_quiz/domains/mbti/scoring.py` — sign of `μ` per dimension determines letter; magnitude maps to preference strength

### Schema — IPIP-NEO 8-facet subset (80 items)

The active `mbti.json` uses an IPIP-NEO-derived item bank, not the original hand-authored 30 questions. See the Decision Log below for rationale.

**Critical weight convention — Agreeableness inversion:**
High Agreeableness = Feeling. In the engine's weight convention (`TF` positive pole = T):
- `+keyed` A items (agree = more agreeable = Feeling) → `weight[TF] = -1.0`
- `-keyed` A items (agree = less agreeable = Thinking) → `weight[TF] = +1.0`

The `agree_indicates_mbti` field in the source JSON encodes this. Never recompute from `keying` alone.

---

## REST API

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/start_quiz` | Start session; returns first question |
| `POST` | `/answer` | Submit response; returns next question or done |
| `GET` | `/result?session_id=<id>` | MBTI type + dimension scores |

`GET /` serves `frontend/index.html` directly (no separate frontend server needed).

**CORS:** Configurable via `CORS_ORIGINS` env var (comma-separated). Defaults to `http://localhost:3000,http://localhost:8000`.

**Sessions:** `session_store.py` auto-selects `MemorySessionStore` (dev, lost on restart) or `SupabaseSessionStore` (prod) based on `SUPABASE_URL` / `SUPABASE_KEY` env vars.

**Vercel deployment:** Entry point is `api/index.py`; `vercel.json` rewrites all routes to it. Requires `SUPABASE_URL` and `SUPABASE_KEY` set in Vercel env.

---

## Extending the Engine

### New domain
1. `adaptive_quiz/domains/<domain>/schema.py` — load questions with weight vectors
2. `adaptive_quiz/domains/<domain>/scoring.py` — map posterior → result
3. Register in `backend/schemas.py`; core engine needs no changes

### New selection strategy / stopping rule / noise model
Implement the respective Protocol from `selection.py` / `stopping.py` / `engine.py` and pass to `AdaptiveEngine(...)`.

---

## Testing

`test_modules.py` (unittest). Key classes:
- `TestBayesianUpdate` — uncertainty reduction, pure function behavior
- `TestAdaptiveQuestionSelector` — weight generation, selection modes
- `TestSyntheticUser` — response simulation, archetypes

`synthetic_user.py` archetypes: `ConfidentUser` (low noise, extreme), `HesitantUser` (high noise), `FragmentedUser` (flip_probability param).

---

## Known Limitations

- **No persistence:** Sessions lost on restart.
- **Response time ignored in API:** `backend/main.py` hardcodes `response_time=1.0`; frontend does not capture it.
- **No error handling on invalid session IDs:** Will raise `KeyError` in session store.

---

## Decision Log

### 2026-03-22 — Schema replacement: 30-item ad-hoc → 80-item IPIP-NEO subset

**Context:** The original `mbti.json` was hand-authored with empirically unfounded cross-loadings and no validated psychometric source.

**Options considered:**

| Option | Items | Decision |
|--------|-------|----------|
| Keep existing schema | 30 | Rejected — no validation, incorrect cross-loadings |
| IPIP Big Five Factor Markers (Goldberg 1992) | 50 | Rejected — 10 Neuroticism items have no MBTI counterpart; adaptive engine wastes question budget on them |
| IPIP-NEO 8-facet subset (Goldberg 1999) | 80 | **Selected** |
| IPIP-NEO full instrument | 300 | Rejected — 60 Neuroticism items, political content (O6), eating-disorder-adjacent content (N5) |

**The 8 facets selected:**

| MBTI | Facets | Rationale |
|------|--------|-----------|
| EI | E1 Friendliness (α=.87), E2 Gregariousness (α=.79) | Purest social-preference facets |
| NS | O1 Imagination (α=.83), O5 Intellect (α=.86) | Abstract/concrete cognitive style; avoids political (O6) and TF bleed (O3) |
| TF | A3 Altruism (α=.77), A6 Sympathy (α=.75) | Decision-making warmth/empathy; other A facets measure interpersonal orientation |
| JP | C2 Orderliness (α=.82), C5 Self-Discipline (α=.85) | Strongest JP proxies; C3/C4 load on work ethic |

**Source files:** `adaptive_quiz/schemas/files_extracted/` — IPIP JSON item banks and explainer. Public domain: https://ipip.ori.org

---

## Next Steps

### High priority

- **IRT calibration** — Current weight vectors are uniform (±1.0). Fit a 2PL IRT model to the Open Psychometrics IPIP-NEO dataset (~1M responses, https://openpsychometrics.org/_rawdata/) and replace with empirically derived *a*-parameters. Highest-leverage improvement to inference accuracy.

- **Response scale alignment** — IPIP uses 1–5 Likert. Verify frontend presents IPIP standard labels: *Very Inaccurate / Moderately Inaccurate / Neither / Moderately Accurate / Very Accurate* (not agree/disagree framing).

- **Item ordering** — Schema currently groups items by facet. Shuffle/interleave across dimensions per session to reduce acquiescence bias.

### Medium priority

- **Neuroticism bonus output** — N1 Anxiety (α=.83) + N3 Depression (α=.88) facets from the 300-item file could score Neuroticism as a 5th supplementary output after MBTI convergence.

- **Session persistence** — Swap `session_store.py` implementation for SQLite or Redis.

- **Response time in API** — Wire frontend wall-clock time into `/answer` payload; noise model already accepts it.

### Lower priority

- **Big Five output mode** — Outputting raw E/O/A/C scores without the MBTI mapping layer is a one-line change in `scoring.py`.

- **IPIP-NEO-120** — Johnson (2014) 4-highest-discriminating-items-per-facet selection gives a 32-item bank for the 8-facet subset — leaner if convergence studies show current 80 items are redundant.

- **Cross-loadings** — Current schema has zero cross-loadings. Introducing empirically validated cross-loadings from published factor analyses would give richer per-question information at the cost of weight convention complexity.
