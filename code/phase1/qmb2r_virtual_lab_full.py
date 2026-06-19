#!/usr/bin/env python3
"""QMB-2R Virtual Quantum Laboratory — reproducible Phase 1 runner.

Regenerates the baseline qubit scan, damage-family robustness ensemble,
observable-dependence test, three-level transmon leakage scan, memory-ancilla
scan, scaling fit and blind synthetic experiment.

The program uses standard open-system quantum mechanics. QMB-2R is treated as
a hypothesis about boundary-regulated retention of a chosen observable
identity, not as a replacement for quantum mechanics.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

SEED = 20260619
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_reproduced_phase1"
FIG = ROOT / "figures_reproduced_phase1"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
sm = np.array([[0, 1], [0, 0]], complex)
sp = sm.conj().T
I2 = np.eye(2, dtype=complex)


def ket(index: int, dimension: int) -> np.ndarray:
    vector = np.zeros((dimension, 1), complex)
    vector[index, 0] = 1
    return vector


def density(index: int, dimension: int) -> np.ndarray:
    vector = ket(index, dimension)
    return vector @ vector.conj().T


def liouvillian(hamiltonian: np.ndarray, collapse_ops: list[np.ndarray]) -> np.ndarray:
    dimension = hamiltonian.shape[0]
    identity = np.eye(dimension, dtype=complex)
    generator = -1j * (
        np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity)
    )
    for operator in collapse_ops:
        product = operator.conj().T @ operator
        generator += (
            np.kron(operator.conj(), operator)
            - 0.5 * np.kron(identity, product)
            - 0.5 * np.kron(product.T, identity)
        )
    return generator


def evolve(hamiltonian, collapse_ops, initial_state, time):
    dimension = hamiltonian.shape[0]
    vector = expm(liouvillian(hamiltonian, collapse_ops) * time) @ initial_state.reshape(
        -1, order="F"
    )
    state = vector.reshape((dimension, dimension), order="F")
    return 0.5 * (state + state.conj().T)


def trace_distance(first, second) -> float:
    difference = first - second
    difference = 0.5 * (difference + difference.conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def eigenstate_density(operator, positive: bool):
    values, vectors = np.linalg.eigh(operator)
    index = int(np.argmax(values) if positive else np.argmin(values))
    vector = vectors[:, index : index + 1]
    return vector @ vector.conj().T


def qubit_metrics(
    kappa: float,
    omega: float = 1.0,
    gamma1: float = 0.05,
    gamma_phi: float = 0.02,
    eta_down: float = 0.015,
    eta_up: float = 0.005,
) -> tuple[float, float]:
    time = np.pi / (2 * omega)
    gamma_population = gamma1 + (eta_down + eta_up) * kappa**2
    gamma_transverse = gamma_population / 2 + gamma_phi + kappa
    matrix = np.array(
        [[-gamma_transverse, omega], [-omega, -gamma_population]], float
    )
    difference = expm(matrix * time) @ np.array([0.0, 2.0])
    return float(abs(difference[1]) / 2), float(np.linalg.norm(difference) / 2)


def qutrit_result(
    kappa: float,
    omega: float = 1.0,
    anharmonicity: float = -5.0,
    gamma1: float = 0.04,
    gamma_phi: float = 0.015,
    eta_leak: float = 0.008,
    eta_heat: float = 0.002,
    detuning: float = 0.0,
) -> dict:
    dimension = 3
    time = np.pi / (2 * omega)
    lowering = np.zeros((dimension, dimension), complex)
    for level in range(1, dimension):
        lowering[level - 1, level] = np.sqrt(level)
    raising = lowering.conj().T
    number = raising @ lowering
    hamiltonian = (
        0.5 * omega * (lowering + raising)
        + 0.5 * anharmonicity * (number @ (number - np.eye(dimension)))
        + detuning * number
    )
    collapse_ops = [
        np.sqrt(gamma1) * lowering,
        np.sqrt(2 * (gamma_phi + kappa)) * number,
    ]
    leakage = np.zeros((dimension, dimension), complex)
    leakage[2, 1] = 1
    heating = np.zeros((dimension, dimension), complex)
    heating[1, 0] = 1
    if eta_leak * kappa**2 > 0:
        collapse_ops.append(np.sqrt(eta_leak * kappa**2) * leakage)
    if eta_heat * kappa**2 > 0:
        collapse_ops.append(np.sqrt(eta_heat * kappa**2) * heating)
    state_zero = evolve(hamiltonian, collapse_ops, density(0, dimension), time)
    state_one = evolve(hamiltonian, collapse_ops, density(1, dimension), time)
    computational_z = np.diag([1.0, -1.0, 0.0])
    probability_one_zero = float(state_zero[1, 1].real)
    probability_one_one = float(state_one[1, 1].real)
    return {
        "computational_identity_Cz": float(
            abs(np.trace(computational_z @ (state_zero - state_one)).real) / 2
        ),
        "full_trace_distance": trace_distance(state_zero, state_one),
        "binary_readout_contrast": abs(probability_one_one - probability_one_zero),
        "average_leakage": float(
            (state_zero[2, 2].real + state_one[2, 2].real) / 2
        ),
        "p1_from_zero": probability_one_zero,
        "p1_from_one": probability_one_one,
    }


def partial_trace_environment(state):
    return np.trace(state.reshape(2, 2, 2, 2), axis1=1, axis2=3)


def memory_metrics(
    kappa: float,
    omega: float = 1.0,
    coupling: float = 0.35,
    environment_decay: float = 0.12,
    gamma1: float = 0.03,
    gamma_phi: float = 0.01,
    eta_damage: float = 0.012,
) -> tuple[float, float]:
    time = np.pi / (2 * omega)
    hamiltonian = (
        0.5 * omega * np.kron(sx, I2)
        + coupling * (np.kron(sp, sm) + np.kron(sm, sp))
    )
    collapse_ops = [
        np.sqrt(gamma1 + eta_damage * kappa**2) * np.kron(sm, I2),
        np.sqrt((gamma_phi + kappa) / 2) * np.kron(sz, I2),
        np.sqrt(environment_decay) * np.kron(I2, sm),
    ]
    environment = density(0, 2)
    state_zero = evolve(
        hamiltonian, collapse_ops, np.kron(density(0, 2), environment), time
    )
    state_one = evolve(
        hamiltonian, collapse_ops, np.kron(density(1, 2), environment), time
    )
    reduced_zero = partial_trace_environment(state_zero)
    reduced_one = partial_trace_environment(state_one)
    computational_identity = abs(
        np.trace(sz @ (reduced_zero - reduced_one)).real
    ) / 2
    return float(computational_identity), trace_distance(reduced_zero, reduced_one)


def save_plot(x, ys, labels, title, xlabel, ylabel, path, ylim=None):
    plt.figure(figsize=(8, 5))
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


# 1. Baseline
grid = np.linspace(0, 8, 241)
baseline = pd.DataFrame(
    [
        {
            "kappa_over_omega": kappa,
            "computational_identity_Cz": qubit_metrics(kappa)[0],
            "full_trace_distance": qubit_metrics(kappa)[1],
        }
        for kappa in grid
    ]
)
baseline.to_csv(DATA / "baseline_qubit_scan.csv", index=False)
save_plot(
    grid,
    [baseline.computational_identity_Cz, baseline.full_trace_distance],
    ["Computational-basis identity", "Full trace distance"],
    "Baseline QMB-2R qubit model",
    "Boundary coupling kappa / Omega",
    "Retained distinguishability",
    FIG / "01_baseline_qubit.png",
    (0, 1.05),
)

# 2. Damage-family robustness
records = []
for family in ["none", "linear", "quadratic", "threshold"]:
    for sample in range(250):
        omega = float(np.exp(RNG.uniform(np.log(0.5), np.log(2.0))))
        gamma1 = float(RNG.uniform(0.005, 0.10) * omega)
        gamma_phi = float(RNG.uniform(0.0, 0.06) * omega)
        eta = float(np.exp(RNG.uniform(np.log(0.005), np.log(0.10))))
        threshold = float(RNG.uniform(0.5, 2.0) * omega)

        def damage(kappa: float) -> float:
            if family == "none":
                return 0.0
            if family == "linear":
                return eta * kappa
            if family == "quadratic":
                return eta * kappa**2 / omega
            return eta * max(kappa - threshold, 0.0) ** 2 / omega

        scan = np.linspace(0, 10 * omega, 151)
        time = np.pi / (2 * omega)
        curve = []
        for kappa in scan:
            population = gamma1 + damage(kappa)
            transverse = population / 2 + gamma_phi + kappa
            matrix = np.array([[-transverse, omega], [-omega, -population]], float)
            difference = expm(matrix * time) @ np.array([0.0, 2.0])
            curve.append(abs(difference[1]) / 2)
        curve = np.asarray(curve)
        index = int(np.argmax(curve))
        gain = curve[index] - max(curve[0], curve[-1])
        records.append(
            {
                "family": family,
                "sample": sample,
                "optimal_kappa_over_omega": scan[index] / omega,
                "optimal_Cz": curve[index],
                "finite_optimum_pass": bool(
                    0 < index < len(scan) - 1 and gain > 0.02
                ),
            }
        )
family_data = pd.DataFrame(records)
family_data.to_csv(DATA / "damage_family_robustness_samples.csv", index=False)
family_data.groupby("family").agg(
    trials=("sample", "count"),
    finite_optimum_rate=("finite_optimum_pass", "mean"),
    median_optimal_kappa_over_omega=("optimal_kappa_over_omega", "median"),
).reset_index().to_csv(DATA / "damage_family_summary.csv", index=False)

# 3. Observable dependence
axis_rows = []
for name, observable in {"sigma_x": sx, "sigma_y": sy, "sigma_z": sz}.items():
    for kappa in np.linspace(0, 8, 161):
        hamiltonian = 0.5 * sx
        collapse_ops = [
            np.sqrt(0.05 + 0.015 * kappa**2) * sm,
            np.sqrt(0.005 * kappa**2) * sp,
            np.sqrt((0.02 + kappa) / 2) * sz,
        ]
        positive = evolve(
            hamiltonian, collapse_ops, eigenstate_density(observable, True), np.pi / 2
        )
        negative = evolve(
            hamiltonian,
            collapse_ops,
            eigenstate_density(observable, False),
            np.pi / 2,
        )
        axis_rows.append(
            {
                "observable": name,
                "kappa_over_omega": kappa,
                "projected_identity": abs(
                    np.trace(observable @ (positive - negative)).real
                )
                / 2,
            }
        )
pd.DataFrame(axis_rows).to_csv(DATA / "observable_dependence.csv", index=False)

# 4. Qutrit
qutrit = pd.DataFrame(
    [
        {"kappa_over_omega": kappa, **qutrit_result(kappa)}
        for kappa in np.linspace(0, 20, 201)
    ]
)
qutrit.to_csv(DATA / "qutrit_leakage_scan.csv", index=False)
save_plot(
    qutrit.kappa_over_omega,
    [qutrit.computational_identity_Cz, qutrit.average_leakage],
    ["Computational identity", "Leakage"],
    "Three-level transmon test",
    "Boundary coupling kappa / Omega",
    "Probability-scale metric",
    FIG / "04_qutrit_leakage.png",
    (0, 1.05),
)

# 5. Memory ancilla
memory = pd.DataFrame(
    [
        {
            "kappa_over_omega": kappa,
            "computational_identity_Cz": memory_metrics(kappa)[0],
            "full_trace_distance": memory_metrics(kappa)[1],
        }
        for kappa in np.linspace(0, 10, 121)
    ]
)
memory.to_csv(DATA / "memory_ancilla_scan.csv", index=False)

# 6. Scaling
scaling = []
for trial in range(400):
    omega = float(np.exp(RNG.uniform(np.log(0.4), np.log(2.5))))
    eta = float(np.exp(RNG.uniform(np.log(0.002), np.log(0.05))) / omega)
    result = minimize_scalar(
        lambda kappa: -qubit_metrics(
            kappa,
            omega=omega,
            gamma1=0.02 * omega,
            gamma_phi=0.01 * omega,
            eta_down=eta,
            eta_up=0.25 * eta,
        )[0],
        bounds=(0.02 * omega, 10 * omega),
        method="bounded",
    )
    scaling.append({"omega": omega, "eta": eta, "optimal_kappa": result.x})
scaling = pd.DataFrame(scaling)
design = np.column_stack(
    [np.ones(len(scaling)), np.log(scaling.omega), np.log(scaling.eta)]
)
target = np.log(scaling.optimal_kappa)
coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
predicted = design @ coefficients
r_squared = 1 - np.sum((target - predicted) ** 2) / np.sum(
    (target - target.mean()) ** 2
)
scaling.to_csv(DATA / "scaling_law_samples.csv", index=False)
(DATA / "scaling_fit.json").write_text(
    json.dumps(
        {
            "intercept": float(coefficients[0]),
            "omega_exponent": float(coefficients[1]),
            "eta_exponent": float(coefficients[2]),
            "r_squared": float(r_squared),
        },
        indent=2,
    )
)

# 7. Blind virtual experiment
blind_grid = np.linspace(0, 10, 21)
predictor = {
    "omega": 1.0,
    "anharmonicity": -5.0,
    "gamma1": 0.04,
    "gamma_phi": 0.015,
    "eta_leak": 0.008,
    "eta_heat": 0.002,
    "detuning": 0.0,
}
hidden = {
    "omega": 1.02,
    "anharmonicity": -4.7,
    "gamma1": 0.043,
    "gamma_phi": 0.017,
    "eta_leak": 0.0092,
    "eta_heat": 0.0023,
    "detuning": 0.03,
}
prediction_curve = np.array(
    [qutrit_result(kappa, **predictor)["binary_readout_contrast"] for kappa in blind_grid]
)
predicted_optimum = blind_grid[int(np.argmax(prediction_curve))]
blind_rows = []
shots = 20_000
assignment_error = 0.02
for kappa, prediction in zip(blind_grid, prediction_curve):
    truth = qutrit_result(kappa, **hidden)
    p0 = assignment_error + (1 - 2 * assignment_error) * truth["p1_from_zero"]
    p1 = assignment_error + (1 - 2 * assignment_error) * truth["p1_from_one"]
    count0 = RNG.binomial(shots, p0)
    count1 = RNG.binomial(shots, p1)
    corrected0 = (count0 / shots - assignment_error) / (1 - 2 * assignment_error)
    corrected1 = (count1 / shots - assignment_error) / (1 - 2 * assignment_error)
    blind_rows.append(
        {
            "kappa_over_omega": kappa,
            "predicted_contrast": prediction,
            "hidden_true_contrast": truth["binary_readout_contrast"],
            "observed_contrast": abs(corrected1 - corrected0),
        }
    )
blind = pd.DataFrame(blind_rows)
blind.to_csv(DATA / "blind_synthetic_experiment.csv", index=False)
observed_optimum = float(
    blind.loc[blind.observed_contrast.idxmax(), "kappa_over_omega"]
)

summary = {
    "seed": SEED,
    "baseline_optimum": float(
        baseline.loc[
            baseline.computational_identity_Cz.idxmax(), "kappa_over_omega"
        ]
    ),
    "qutrit_optimum": float(
        qutrit.loc[qutrit.computational_identity_Cz.idxmax(), "kappa_over_omega"]
    ),
    "memory_optimum": float(
        memory.loc[memory.computational_identity_Cz.idxmax(), "kappa_over_omega"]
    ),
    "scaling_omega_exponent": float(coefficients[1]),
    "scaling_eta_exponent": float(coefficients[2]),
    "scaling_r_squared": float(r_squared),
    "blind_predicted_optimum": float(predicted_optimum),
    "blind_observed_optimum": observed_optimum,
}
(DATA / "reproduced_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
