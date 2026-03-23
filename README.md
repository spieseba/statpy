# statpy

A Python toolkit for statistical analysis of Markov Chain Monte Carlo data in the context of lattice QCD. These tools were created during the course of my PhD work, including applications like the short-distance window HVP analysis. An applied example can be found in the [correlatorCAA](https://github.com/spieseba/correlatorCAA) repository.

The toolset primarily offers robust statistical methods for analyzing large sets of observables across various ensembles, while remaining flexible enough to support a wide range of statistical tasks.

---

## Getting started

### Prerequisites
- Python >= 3.12
- `uv` installed (see https://docs.astral.sh/uv/getting-started/)

### Regular install (non-editable)
Use this if you just want to **use** `statpy` without modifying it.

From your project directory:

```bash
uv venv --python 3.12
uv pip install /absolute/path/to/statpy
```

### Development install (editable)
Use this if you actively develop statpy and want changes
to be picked up immediately by consuming projects.

From your project directory:

```bash
uv venv --python 3.12
uv pip install meson meson-python ninja
uv add --editable /absolute/path/to/statpy                       # Record the dependency (so version constraints are resepected)
uv pip install -e /absolute/path/to/statpy --no-build-isolation  # Reinstall using project environment for build tools
```
The build tools are installed into the project environment and reused for editable builds.

---

### Verify installation
```bash
uv run python -c "import statpy; print(statpy.__file__)"
```
