# Optimum-Centred Protection–Damage Priors for Quantum Calibration

This repository accompanies the manuscript:

**Optimum-Centred Protection–Damage Priors for Sample-Efficient Quantum Calibration Under Drift, Leakage, and Incomplete Metadata**

**Author:** Abubakar Rabiu Salihawa  
**Affiliation:** Independent Researcher, Nigeria  
**Contact:** abubakarsalihawa019@gmail.com

## Scope

The project studies an interpretable prior for adaptive calibration of open quantum systems. It does **not** claim a new law of physics, a modification of quantum mechanics, or experimental verification. All reported results are computational and use standard open-system quantum dynamics and statistical learning.

## Main finding

An optimum-centred protection–damage prior can improve low-budget calibration on related virtual devices. The advantage weakens under hardware realism and can become unsafe under severe nonstationarity unless residual correction and prior-rejection mechanisms are used.

## Repository contents

- `code/` — available simulation and analysis source.
- `data/` — consolidated results and selected frozen summaries.
- `paper/` — bibliography; manuscript files are distributed with the archived release.
- `docs/` — disclosure, contribution guidance, and release notes.
- `CITATION.cff` — citation metadata.
- `.zenodo.json` — archival metadata for the first DOI release.

## Reproduce manuscript figures

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python code/analysis/reproduce_manuscript_figures.py
```

## Reproducibility status

The repository contains frozen result tables and the complete available source for Phases 1 and 2. Historical interactive development code for some later virtual-laboratory phases is not yet consolidated into one clean end-to-end runner; this limitation is disclosed in the manuscript and supplementary information. No result should be treated as experimentally validated.

## Citation

Use `CITATION.cff`, or cite the software and manuscript DOI after the first archived release.

## Licensing

- Source code: MIT License (`LICENSE`).
- Manuscript, figures, documentation, and datasets: CC BY 4.0 (`LICENSE-CONTENT.md`).
