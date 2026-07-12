# statpy
[![tests](https://github.com/spieseba/statpy/actions/workflows/ci.yml/badge.svg)](https://github.com/spieseba/statpy/actions/workflows/ci.yml)

A Python toolkit for statistical analysis of Markov Chain Monte Carlo data in the context of lattice QCD.

## What it does

- **Automatic error propagation through arbitrary, nonlinear operations:** `db.transform(tag, f)` and `db.combine(t1, t2, f=...)` apply any function to the central value and to every jackknife/bootstrap resample, so error bars on quantities like effective masses `log(C(t)/C(t+1))` or ratios fall out with no hand-derived Jacobians and no Gaussian linearization.
- **Correlations preserved by construction:** combining entries aligns jackknife resamples by config label (union of config sets, configs missing from an input contributing their central value) and bootstrap resamples by index, so covariances between derived quantities are carried through every step.
- **Autocorrelation-aware UQ for MCMC time series:** jackknife, bootstrap, and a published delayed-binning estimator ([Phys. Rev. D 111](https://doi.org/10.1103/mj3d-yq87)) for errors on correlated samples, plus correlated least-squares fitting with the full covariance matrix.
- **Reproducible, self-describing storage:** a tagged binary database carries samples, jackknife blocks, and the statpy commit hash in a CRC-checked file, so an analysis and its provenance travel together.

```python
import numpy as np
import statpy as sp

rng = np.random.default_rng(0)
n_cfg, n_t = 200, 8
cfgs = np.array([f"cfg-{i}" for i in range(n_cfg)])
w = np.ones(n_cfg)

# Two measured correlators, one sample per configuration.
t = np.arange(n_t)
C1 = np.exp(-0.5 * t) * (1 + 0.05 * rng.normal(size=(n_cfg, n_t)))
C2 = np.exp(-0.7 * t) * (1 + 0.05 * rng.normal(size=(n_cfg, n_t)))

db = sp.database.core.DB()
db.add_entry("C1", sample=C1, weights=w, cfgs=cfgs)
db.add_entry("C2", sample=C2, weights=w, cfgs=cfgs)

# Transform one entry: effective mass  m(t) = log(C1(t) / C1(t+1)).
db.transform("C1", lambda c: np.log(c[:-1] / c[1:]), store_as="m_eff")

# Combine two entries cfg-wise: the ratio C1 / C2.
db.combine("C1", "C2", f=lambda a, b: a / b, store_as="ratio")

# Jackknife errors propagate automatically through both operations.
err = lambda tag: np.sqrt(db.jackknife_variance(tag))
print("m_eff:", db.database["m_eff"].central_value[:3].round(3), "+/-", err("m_eff")[:3].round(3))
print("ratio:", db.database["ratio"].central_value[:3].round(3), "+/-", err("ratio")[:3].round(3))
# m_eff: [0.505 0.5   0.494] +/- [0.005 0.005 0.005]
# ratio: [1.001 1.225 1.491] +/- [0.005 0.006 0.008]
```

See [caa-control-variates](https://github.com/spieseba/caa-control-variates) for a full worked example (Monte Carlo variance reduction on real data).

---

### Prerequisites
- Python >= 3.12
- `uv` installed (see https://docs.astral.sh/uv/getting-started/)

### Installation
From your project directory, run **one** of:

```bash
uv add /absolute/path/to/statpy             # non-editable
uv add --editable /absolute/path/to/statpy  # editable
```

Use the editable command if you actively develop statpy and want changes
to be picked up immediately by consuming projects.

statpy is pure Python with a `uv_build` backend, so the editable install
is a plain path link — source edits are picked up live with no rebuild step.

### Verify installation
```bash
uv run python -c "import statpy; print(statpy.__file__)"
```

---

### Versions
This is the actively developed version (v2). The original toolkit, written during my PhD, is preserved as the [v1.0 release](https://github.com/spieseba/statpy/releases/tag/v1.0) and on the [`v1-legacy`](https://github.com/spieseba/statpy/tree/v1-legacy) branch, and is no longer maintained.

v2 is a ground-up rewrite and ships a migrator (`statpy.database.io.load_v1_json`) for the retired v1 database format; a round-trip test checks that the migrated v2 entries reproduce the v1 means and jackknife blocks to machine precision.
