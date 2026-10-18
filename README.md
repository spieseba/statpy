# statpy
[![tests](https://github.com/spieseba/statpy/actions/workflows/ci.yml/badge.svg)](https://github.com/spieseba/statpy/actions/workflows/ci.yml)

A Python toolkit for statistical analysis of Markov Chain Monte Carlo data in the context of lattice QCD.

## What it does

- **Automatic error propagation through arbitrary, nonlinear operations:** `Variable.derive(f, *variables)` applies any function to the central value and to every jackknife/bootstrap resample, so error bars on quantities like effective masses `log(C(t)/C(t+1))` or ratios fall out with no hand-derived Jacobians and no Gaussian linearization.
- **Correlations preserved by construction:** combining variables aligns jackknife resamples by config label (union of config sets, configs missing from an input contributing their central value) and bootstrap resamples by index, so covariances between derived quantities are carried through every step.
- **Autocorrelation-aware UQ for MCMC time series:** jackknife, bootstrap, and a published delayed-binning estimator ([Phys. Rev. D 111](https://doi.org/10.1103/mj3d-yq87)) for errors on correlated samples, plus correlated least-squares fitting with the full covariance matrix.
- **Reproducible, self-describing storage:** a tagged binary database carries observations, jackknife replicates, and the statpy commit hash in a CRC-checked file, so an analysis and its provenance travel together.

```python
import numpy as np
import statpy as sp
from statpy.variable import Layout, Variable

rng = np.random.default_rng(0)
num_cfg, num_t = 200, 8
layout = Layout("A1", np.arange(num_cfg))  # ensemble, Monte Carlo times

# Two measured correlators, one sample per configuration.
t = np.arange(num_t)
C1 = np.exp(-0.5 * t) * (1 + 0.05 * rng.normal(size=(num_cfg, num_t)))
C2 = np.exp(-0.7 * t) * (1 + 0.05 * rng.normal(size=(num_cfg, num_t)))

db = sp.database.Database()
db["C1"] = Variable.from_data(C1, layout)
db["C2"] = Variable.from_data(C2, layout)

# Transform one variable: effective mass  m(t) = log(C1(t) / C1(t+1)).
db["m_eff"] = Variable.derive(lambda c: np.log(c[:-1] / c[1:]), db["C1"])

# Combine two variables cfg-wise: the ratio C1 / C2.
db["ratio"] = Variable.derive(lambda a, b: a / b, db["C1"], db["C2"])

# Jackknife errors propagate automatically through both operations.
err = lambda tag: np.sqrt(db[tag].jackknife_variance())
print("m_eff:", db["m_eff"].central_value[:3].round(3), "+/-", err("m_eff")[:3].round(3))
print("ratio:", db["ratio"].central_value[:3].round(3), "+/-", err("ratio")[:3].round(3))
# m_eff: [0.505 0.5   0.494] +/- [0.005 0.005 0.005]
# ratio: [1.001 1.225 1.491] +/- [0.005 0.006 0.008]
```

See [caa-control-variates](https://github.com/spieseba/caa-control-variates) for a full worked example (Monte Carlo variance reduction on real data).

---

### Prerequisites
- Python >= 3.12 (developed and tested with 3.14; older versions are no longer checked)
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
