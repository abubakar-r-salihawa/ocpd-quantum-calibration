#!/usr/bin/env python3
"""Reproduce manuscript result figures from frozen CSV outputs.

Run from any directory after installing ``requirements.txt``:

    python code/analysis/reproduce_manuscript_figures.py

The theorem figure is generated analytically. Other figures are reproduced
from frozen trial-level outputs deposited with the archived research release.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "figures_reproduced"
OUT.mkdir(exist_ok=True)
DATA_ROOT = REPO_ROOT / "data" / "frozen-phases"

P3_PATH = DATA_ROOT / "phase3_data" / "phase3_all_trials.csv"
P5_PATH = DATA_ROOT / "phase5_data" / "phase5_all_trials.csv"
P6_PATH = DATA_ROOT / "phase6_data" / "phase6_all_trials.csv"

for path in (P3_PATH, P5_PATH, P6_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen trial data: {path}. Download the full archived "
            "release and extract its data/frozen-phases directory."
        )

P3 = pd.read_csv(P3_PATH)
P5 = pd.read_csv(P5_PATH)
P6 = pd.read_csv(P6_PATH)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


# Figure 1: theorem
kappa = np.logspace(-1, 1, 400)
A, B, m, n = 1.0, 0.08, 1.0, 2.0
protection = A * kappa ** (-m)
damage = B * kappa**n
loss = protection + damage
kappa_star = (m * A / (n * B)) ** (1 / (m + n))
plt.figure(figsize=(7.2, 4.8))
plt.plot(kappa, protection, label="Protection deficit")
plt.plot(kappa, damage, label="Boundary damage")
plt.plot(kappa, loss, label="Total loss", linewidth=2)
plt.axvline(kappa_star, linestyle="--", label="Optimum")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Boundary coupling")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "figure1_theorem.png", dpi=300)
plt.close()

# Figure 2: Phase 3
primary = ["Exact power-law", "Perturbed power-law", "Driven qubit"]
items = [
    ("QMB-only", "OCPD-only"),
    ("Hybrid QMB+GP", "OCPD+GP"),
    ("Bayesian", "Bayesian"),
    ("Golden", "Golden"),
    ("Grid", "Grid"),
]
values, errors, labels = [], [], []
for key, label in items:
    group = P3[P3.family.isin(primary) & (P3.budget == 8) & (P3.method == key)]
    p, low, high = wilson(int(group.within_2_percent_of_optimum.sum()), len(group))
    values.append(100 * p)
    errors.append([100 * (p - low), 100 * (high - p)])
    labels.append(label)
plt.figure(figsize=(7.2, 4.8))
x = np.arange(len(labels))
plt.bar(x, values, yerr=np.array(errors).T, capsize=4)
plt.xticks(x, labels, rotation=20, ha="right")
plt.ylabel("Near-optimal success (%)")
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "figure2_phase3_methods.png", dpi=300)
plt.close()

# Figure 3: Phase 5
plt.figure(figsize=(7.2, 4.8))
for key, label in [
    ("Gated QMB+GP", "Gated OCPD+GP"),
    ("Direct metadata+GP", "Direct metadata+GP"),
    ("Fresh Bayesian", "Fresh Bayesian"),
    ("Grid", "Grid"),
]:
    group = (
        P5[(P5.scenario == "Related") & (P5.method == key)]
        .groupby("budget")
        .success_98pct.mean()
        .reset_index()
    )
    plt.plot(group.budget, 100 * group.success_98pct, marker="o", label=label)
plt.axhline(90, linestyle="--")
plt.xlabel("Measured boundary settings")
plt.ylabel("Near-optimal success (%)")
plt.ylim(0, 105)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "figure3_phase5_transfer.png", dpi=300)
plt.close()

# Figure 4: Phase 6 moderate realism
plt.figure(figsize=(7.2, 4.8))
for key, label in [
    ("Gated robust QMB+GP", "Gated robust OCPD+GP"),
    ("Direct robust metadata+GP", "Direct robust metadata+GP"),
    ("Static QMB+GP", "Static OCPD+GP"),
    ("Fresh Bayesian", "Fresh Bayesian"),
]:
    group = (
        P6[(P6.scenario == "Moderate realism") & (P6.method == key)]
        .groupby("budget")
        .success_98pct.mean()
        .reset_index()
    )
    plt.plot(group.budget, 100 * group.success_98pct, marker="o", label=label)
plt.axhline(85, linestyle="--")
plt.xlabel("Measured boundary settings")
plt.ylabel("Near-optimal deployment success (%)")
plt.ylim(0, 105)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "figure4_phase6_moderate.png", dpi=300)
plt.close()

# Figure 5: Phase 6 stress
scenarios = ["Severe drift", "Correlated bursts", "Sparse metadata", "Combined stress"]
group = P6[(P6.budget == 8) & (P6.method == "Gated robust QMB+GP")]
success = [100 * group[group.scenario == s].success_98pct.mean() for s in scenarios]
gate = [100 * group[group.scenario == s].gate_rejected.mean() for s in scenarios]
catastrophic = [
    100 * group[group.scenario == s].catastrophic_below_90pct.mean()
    for s in scenarios
]
x = np.arange(len(scenarios))
width = 0.25
plt.figure(figsize=(8.2, 4.8))
plt.bar(x - width, success, width, label="Near-optimal success")
plt.bar(x, gate, width, label="Gate rejection")
plt.bar(x + width, catastrophic, width, label="Catastrophic selection")
plt.xticks(x, scenarios, rotation=18, ha="right")
plt.ylabel("Rate (%)")
plt.ylim(0, 105)
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "figure5_phase6_stress.png", dpi=300)
plt.close()

print(f"Reproduced figures in {OUT}")
