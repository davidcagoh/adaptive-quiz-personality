"""
Generate report figures and save to assets/.

Run once:
    python generate_assets.py

Outputs:
    assets/convergence.png   -- adaptive vs random vs full box plot (N=500)
    assets/pareto.png        -- threshold sweep accuracy-efficiency Pareto curve
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
ASSETS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def run_convergence(n_users=500):
    from adaptive_quiz.core import (
        AdaptiveEngine, SignConfidenceStopping, VarianceSelection, VarianceThresholdStopping,
    )
    from adaptive_quiz.domains.mbti import load_mbti_schema, mbti_from_posterior
    from adaptive_quiz.simulation.synthetic_user import SyntheticUser

    schema = load_mbti_schema()
    questions = schema["questions"]
    rng_latent = np.random.RandomState(42)
    rng_order = np.random.RandomState(43)

    adaptive_q, random_q = [], []
    axes = ["EI", "NS", "TF", "JP"]
    axis_acc = {ax: [] for ax in axes}

    print(f"Running convergence simulation (N={n_users})...")
    for i in range(n_users):
        user = SyntheticUser(d=4, seed=rng_latent.randint(0, 2**31))
        true_type = mbti_from_posterior(user.theta_true)["type"]

        # Adaptive
        adap = AdaptiveEngine(schema=schema, selection_strategy=VarianceSelection(),
                              stopping_rule=SignConfidenceStopping(confidence_threshold=0.85))
        while not adap.is_complete():
            nq = adap.get_next_question()
            if nq is None:
                break
            qid, w = nq
            y, r, _ = user.simulate_response(w)
            adap.submit_answer(qid, y, r)
        adap_state = adap.get_state()
        adaptive_q.append(adap_state.num_questions)
        adap_type = mbti_from_posterior(adap_state.mu)["type"]
        for j, ax in enumerate(axes):
            axis_acc[ax].append(adap_type[j] == true_type[j])

        # Random order + same stopping rule
        rand = AdaptiveEngine(schema=schema, selection_strategy=VarianceSelection(),
                              stopping_rule=SignConfidenceStopping(confidence_threshold=0.85))
        order = list(range(len(questions)))
        rng_order.shuffle(order)
        asked_ids, rand_n = set(), 0
        for idx in order:
            if rand.is_complete():
                break
            q = questions[idx]
            if q["id"] in asked_ids:
                continue
            y, r, _ = user.simulate_response(np.array(q["weights"]))
            rand.submit_answer(q["id"], y, r)
            asked_ids.add(q["id"])
            rand_n += 1
        random_q.append(rand_n)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_users}")

    return {
        "adaptive": np.array(adaptive_q),
        "random": np.array(random_q),
        "full": len(questions),
        "axis_accuracy": {ax: float(np.mean(v)) for ax, v in axis_acc.items()},
    }


def plot_convergence(data, out):
    adaptive = data["adaptive"]
    random = data["random"]
    full_n = data["full"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    bp = ax1.boxplot(
        [adaptive, random],
        labels=["Adaptive\n(this engine)", "Random order\n+ early stop"],
        widths=0.5, patch_artist=True,
        medianprops=dict(color="white", linewidth=2.5),
        flierprops=dict(marker="o", alpha=0.25, markersize=3),
    )
    colors = ["#4f46e5", "#f59e0b"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color="#777", linewidth=1)

    ax1.axhline(full_n, color="#ef4444", linewidth=1.5, linestyle="--", label=f"Full test ({full_n} q)")
    ax1.set_ylabel("Questions asked")
    ax1.set_title("Questions to convergence  (N=500 simulated users)")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, full_n + 14)

    for i, (vals, color) in enumerate(zip([adaptive, random], colors)):
        med = int(np.median(vals))
        ax1.text(i + 1, med + 2, f"median {med}", ha="center", fontsize=9,
                 color=color, fontweight="bold")

    # Per-axis accuracy
    axis_labels = ["E / I", "N / S", "T / F", "J / P"]
    acc = [data["axis_accuracy"][k] * 100 for k in ["EI", "NS", "TF", "JP"]]
    bars = ax2.bar(axis_labels, acc, color="#4f46e5", alpha=0.8, width=0.5)
    ax2.set_ylim(85, 100)
    ax2.set_ylabel("Accuracy vs. true type (%)")
    ax2.set_title("Per-axis classification accuracy\n(adaptive engine, N=500)")
    for bar, val in zip(bars, acc):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_pareto(out):
    json_path = RESULTS / "threshold_sweep.json"
    if json_path.exists():
        with open(json_path) as f:
            raw = json.load(f)
        thresholds = sorted(raw, key=float)
        mean_q = [raw[k]["mean_questions"] for k in thresholds]
        accuracy = [100 * raw[k]["accuracy_vs_full"] for k in thresholds]
        thresholds = [float(t) for t in thresholds]
    else:
        # Hardcoded from N=1000 run 2026-03-24
        thresholds = [0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
        mean_q     = [24.5,  31.8, 39.9, 47.1, 54.7, 59.1, 65.0]
        accuracy   = [85.4,  86.0, 86.8, 86.8, 86.8, 86.8, 86.8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(thresholds, mean_q, "o-", color="#4f46e5", linewidth=2, markersize=7)
    ax1.set_xlabel("Confidence threshold p")
    ax1.set_ylabel("Mean questions asked")
    ax1.set_title("Questions asked vs. stopping threshold")
    for t, q in zip(thresholds, mean_q):
        ax1.annotate(f"{q:.1f}", (t, q), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    ax2.plot(mean_q, accuracy, "o-", color="#4f46e5", linewidth=2, markersize=7)
    elbow = thresholds.index(0.85)
    ax2.plot(mean_q[elbow], accuracy[elbow], "o", color="#ef4444", markersize=11, zorder=5)
    ax2.annotate(
        f"p=0.85  optimal\n{mean_q[elbow]:.0f}q, {accuracy[elbow]:.1f}%",
        (mean_q[elbow], accuracy[elbow]),
        xytext=(mean_q[elbow] + 3, accuracy[elbow] - 0.35),
        fontsize=8.5, color="#ef4444",
    )
    for t, q, a in zip(thresholds, mean_q, accuracy):
        ax2.annotate(f"  p={t:.2f}", (q, a), fontsize=7.5, color="#555")
    ax2.set_xlabel("Mean questions asked")
    ax2.set_ylabel("Accuracy vs. full test (%)")
    ax2.set_title("Accuracy–efficiency Pareto curve")
    ax2.set_ylim(84, 88.8)

    max_acc = max(accuracy)
    plateau_start = next(q for q, a in zip(mean_q, accuracy) if a >= max_acc - 0.05)
    ax2.axvspan(plateau_start - 1, max(mean_q) + 3, alpha=0.07, color="#ef4444",
                label="Accuracy plateau (schema ceiling)")
    ax2.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    cache = RESULTS / "convergence_data.json"
    if cache.exists():
        print("Loading cached convergence data...")
        with open(cache) as f:
            raw = json.load(f)
        data = {
            "adaptive": np.array(raw["adaptive"]),
            "random": np.array(raw["random"]),
            "full": raw["full"],
            "axis_accuracy": raw["axis_accuracy"],
        }
    else:
        data = run_convergence(n_users=500)
        with open(cache, "w") as f:
            json.dump({
                "adaptive": data["adaptive"].tolist(),
                "random": data["random"].tolist(),
                "full": data["full"],
                "axis_accuracy": data["axis_accuracy"],
            }, f)

    plot_convergence(data, ASSETS / "convergence.png")
    plot_pareto(ASSETS / "pareto.png")
    print("\nDone.")
