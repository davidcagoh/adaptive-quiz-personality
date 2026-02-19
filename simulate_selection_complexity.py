#!/usr/bin/env python3
"""
Selection strategy comparison under increasing complexity:
- Regime A: simple (no mixed loadings, uncorrelated latent)
- Regime B: mixed loadings
- Regime C: mixed loadings + correlated latent
"""

import numpy as np

from adaptive_quiz.core import (
    AdaptiveEngine,
    InformationGainSelection,
    SignConfidenceStopping,
    VarianceSelection,
    VarianceThresholdStopping,
)
from synthetic_schema import generate_synthetic_schema


def sample_latent(n_dims: int, correlation: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
    """
    Sample a latent vector. If correlation == 0, use N(0, I).
    Else use multivariate normal with covariance: diag=1, off-diag=correlation.
    """
    if rng is None:
        rng = np.random.default_rng()
    if correlation == 0.0:
        return rng.standard_normal(n_dims)
    cov = np.full((n_dims, n_dims), correlation)
    np.fill_diagonal(cov, 1.0)
    return rng.multivariate_normal(np.zeros(n_dims), cov)


def simulate_response(
    mu_true: np.ndarray,
    w: np.ndarray,
    noise_sigma: float,
    rng: np.random.Generator,
) -> float:
    """Observed response: w^T mu_true + N(0, noise_sigma^2)."""
    return float(w @ mu_true + rng.standard_normal() * noise_sigma)


def sign_agreement(mu1: np.ndarray, mu2: np.ndarray) -> tuple[bool, np.ndarray]:
    """Overall (all dims match) and per-dimension sign agreement. Uses np.sign (0 -> 0)."""
    s1 = np.sign(mu1)
    s2 = np.sign(mu2)
    per_dim = (s1 == s2).astype(float)
    overall = bool(np.all(s1 == s2))
    return overall, per_dim


def run_full_test(
    mu_true: np.ndarray,
    schema: dict,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run all questions in order; return posterior mean."""
    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=None,
        stopping_rule=VarianceThresholdStopping(variance_threshold=0.001),
        selection_mode="variance",
    )
    for q in schema["questions"]:
        qid = q["id"]
        w = np.asarray(q["weights"])
        r = simulate_response(mu_true, w, noise_sigma, rng)
        engine.submit_answer(qid, r, 1.0)
    return engine.get_state().mu.copy()


def run_adaptive_test(
    mu_true: np.ndarray,
    schema: dict,
    stopping_rule: SignConfidenceStopping,
    selection_strategy,
    noise_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Run adaptive test; return (posterior_mean, num_questions)."""
    engine = AdaptiveEngine(
        schema=schema,
        selection_strategy=selection_strategy,
        stopping_rule=stopping_rule,
    )
    while not engine.is_complete():
        next_q = engine.get_next_question()
        if next_q is None:
            break
        qid, w = next_q
        r = simulate_response(mu_true, w, noise_sigma, rng)
        engine.submit_answer(qid, r, 1.0)
    state = engine.get_state()
    return state.mu.copy(), state.num_questions


def run_regime(
    regime_name: str,
    n_runs: int,
    n_dims: int,
    n_questions: int,
    mixed_loading_strength: float,
    correlation: float,
    noise_sigma: float,
    stopping_rule: SignConfidenceStopping,
    schema_seed: int,
    latent_seed: int,
) -> dict:
    """Run one regime; return metrics for VarianceSelection and InformationGainSelection."""
    schema = {"questions": generate_synthetic_schema(
        n_questions=n_questions,
        n_dims=n_dims,
        mixed_loading_strength=mixed_loading_strength,
        normalize=True,
        random_seed=schema_seed,
    )}
    rng = np.random.default_rng(latent_seed)

    var_questions = []
    var_vs_true = []
    var_per_dim = [[] for _ in range(n_dims)]

    ig_questions = []
    ig_vs_true = []
    ig_per_dim = [[] for _ in range(n_dims)]

    for _ in range(n_runs):
        mu_true = sample_latent(n_dims, correlation, rng)

        mu_full = run_full_test(mu_true, schema, noise_sigma, rng)

        mu_var, nq_var = run_adaptive_test(
            mu_true, schema, stopping_rule, VarianceSelection(), noise_sigma, rng
        )
        ov_var, pd_var = sign_agreement(mu_var, mu_true)
        var_questions.append(nq_var)
        var_vs_true.append(ov_var)
        for d in range(n_dims):
            var_per_dim[d].append(pd_var[d])

        mu_ig, nq_ig = run_adaptive_test(
            mu_true, schema, stopping_rule, InformationGainSelection(), noise_sigma, rng
        )
        ov_ig, pd_ig = sign_agreement(mu_ig, mu_true)
        ig_questions.append(nq_ig)
        ig_vs_true.append(ov_ig)
        for d in range(n_dims):
            ig_per_dim[d].append(pd_ig[d])

    def pct(x):
        return 100 * np.mean(x)

    return {
        "VarianceSelection": {
            "avg_questions": float(np.mean(var_questions)),
            "median_questions": float(np.median(var_questions)),
            "adaptive_vs_true_pct": pct(var_vs_true),
            "per_dim_pct": [pct(var_per_dim[d]) for d in range(n_dims)],
        },
        "InformationGainSelection": {
            "avg_questions": float(np.mean(ig_questions)),
            "median_questions": float(np.median(ig_questions)),
            "adaptive_vs_true_pct": pct(ig_vs_true),
            "per_dim_pct": [pct(ig_per_dim[d]) for d in range(n_dims)],
        },
    }


def main() -> None:
    n_runs = 1000
    n_dims = 4
    n_questions = 40
    noise_var = 0.2
    noise_sigma = np.sqrt(noise_var)
    stopping_rule = SignConfidenceStopping(confidence_threshold=0.95)
    schema_seed = 42
    latent_seed = 123

    print("=" * 50)
    print("Selection Strategy Comparison Under Complexity")
    print(f"Runs: {n_runs}")
    print("Stopping: SignConfidence (0.95)")
    print(f"Noise variance: {noise_var}")
    print("=" * 50)

    # Regime A — Simple
    res_a = run_regime(
        "A",
        n_runs, n_dims, n_questions,
        mixed_loading_strength=0.0,
        correlation=0.0,
        noise_sigma=noise_sigma,
        stopping_rule=stopping_rule,
        schema_seed=schema_seed,
        latent_seed=latent_seed,
    )
    print("\nRegime A — Simple")
    print("-" * 42)
    print("VarianceSelection:")
    print(f"  Avg questions: {res_a['VarianceSelection']['avg_questions']:.2f}")
    print(f"  Median questions: {int(res_a['VarianceSelection']['median_questions'])}")
    print(f"  Adaptive vs TRUE: {res_a['VarianceSelection']['adaptive_vs_true_pct']:.1f}%")
    print("  Per-dim sign vs TRUE:", " ".join(f"D{i}:{res_a['VarianceSelection']['per_dim_pct'][i]:.1f}%" for i in range(n_dims)))
    print("InformationGainSelection:")
    print(f"  Avg questions: {res_a['InformationGainSelection']['avg_questions']:.2f}")
    print(f"  Median questions: {int(res_a['InformationGainSelection']['median_questions'])}")
    print(f"  Adaptive vs TRUE: {res_a['InformationGainSelection']['adaptive_vs_true_pct']:.1f}%")
    print("  Per-dim sign vs TRUE:", " ".join(f"D{i}:{res_a['InformationGainSelection']['per_dim_pct'][i]:.1f}%" for i in range(n_dims)))

    # Regime B — Mixed Loadings
    res_b = run_regime(
        "B",
        n_runs, n_dims, n_questions,
        mixed_loading_strength=0.3,
        correlation=0.0,
        noise_sigma=noise_sigma,
        stopping_rule=stopping_rule,
        schema_seed=schema_seed + 1,
        latent_seed=latent_seed,
    )
    print("\nRegime B — Mixed Loadings")
    print("-" * 42)
    print("VarianceSelection:")
    print(f"  Avg questions: {res_b['VarianceSelection']['avg_questions']:.2f}")
    print(f"  Median questions: {int(res_b['VarianceSelection']['median_questions'])}")
    print(f"  Adaptive vs TRUE: {res_b['VarianceSelection']['adaptive_vs_true_pct']:.1f}%")
    print("  Per-dim sign vs TRUE:", " ".join(f"D{i}:{res_b['VarianceSelection']['per_dim_pct'][i]:.1f}%" for i in range(n_dims)))
    print("InformationGainSelection:")
    print(f"  Avg questions: {res_b['InformationGainSelection']['avg_questions']:.2f}")
    print(f"  Median questions: {int(res_b['InformationGainSelection']['median_questions'])}")
    print(f"  Adaptive vs TRUE: {res_b['InformationGainSelection']['adaptive_vs_true_pct']:.1f}%")
    print("  Per-dim sign vs TRUE:", " ".join(f"D{i}:{res_b['InformationGainSelection']['per_dim_pct'][i]:.1f}%" for i in range(n_dims)))

    # Regime C — Mixed + Correlated
    res_c = run_regime(
        "C",
        n_runs, n_dims, n_questions,
        mixed_loading_strength=0.3,
        correlation=0.3,
        noise_sigma=noise_sigma,
        stopping_rule=stopping_rule,
        schema_seed=schema_seed + 2,
        latent_seed=latent_seed,
    )
    print("\nRegime C — Mixed + Correlated")
    print("-" * 42)
    print("VarianceSelection:")
    print(f"  Avg questions: {res_c['VarianceSelection']['avg_questions']:.2f}")
    print(f"  Median questions: {int(res_c['VarianceSelection']['median_questions'])}")
    print(f"  Adaptive vs TRUE: {res_c['VarianceSelection']['adaptive_vs_true_pct']:.1f}%")
    print("  Per-dim sign vs TRUE:", " ".join(f"D{i}:{res_c['VarianceSelection']['per_dim_pct'][i]:.1f}%" for i in range(n_dims)))
    print("InformationGainSelection:")
    print(f"  Avg questions: {res_c['InformationGainSelection']['avg_questions']:.2f}")
    print(f"  Median questions: {int(res_c['InformationGainSelection']['median_questions'])}")
    print(f"  Adaptive vs TRUE: {res_c['InformationGainSelection']['adaptive_vs_true_pct']:.1f}%")
    print("  Per-dim sign vs TRUE:", " ".join(f"D{i}:{res_c['InformationGainSelection']['per_dim_pct'][i]:.1f}%" for i in range(n_dims)))

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
