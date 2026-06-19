#!/usr/bin/env python3
"""
Use the trained Phase 5 models to generate transferred priors.

Required model files:
- hierarchical_qmb_forest.joblib
- direct_curve_forest.joblib
- curve_pca.joblib

Input metadata order:
omega, gamma1, gamma_phi, eta_down, eta_up
"""

from pathlib import Path
import joblib
import numpy as np

K_MIN = 0.15
K_MAX = 8.0


def metadata_vector(metadata):
    return np.log([
        metadata["omega"],
        metadata["gamma1"],
        metadata["gamma_phi"],
        metadata["eta_down"],
        metadata["eta_up"],
    ])


def reconstruct_qmb_curve(descriptors, x_values):
    kappa_star = np.exp(descriptors[0])
    loss_star = np.exp(descriptors[1])
    curvature = np.exp(descriptors[2])

    kappa = K_MIN * (K_MAX / K_MIN) ** np.asarray(x_values)
    u_value = np.log(kappa / kappa_star)

    loss = loss_star + curvature * (
        2.0 * np.exp(-u_value)
        + np.exp(2.0 * u_value)
        - 3.0
    )

    return np.clip(np.exp(-loss), 1e-5, 0.999)


def load_models(model_directory):
    model_directory = Path(model_directory)
    return {
        "qmb": joblib.load(model_directory / "hierarchical_qmb_forest.joblib"),
        "direct": joblib.load(model_directory / "direct_curve_forest.joblib"),
        "pca": joblib.load(model_directory / "curve_pca.joblib"),
    }


def predict_qmb_prior(models, metadata, x_values):
    features = metadata_vector(metadata).reshape(1, -1)
    descriptor = models["qmb"].predict(features)[0]
    return reconstruct_qmb_curve(descriptor, x_values)


def predict_direct_prior(models, metadata):
    features = metadata_vector(metadata).reshape(1, -1)
    latent = models["direct"].predict(features)
    return np.clip(models["pca"].inverse_transform(latent)[0], 1e-5, 0.999)
