# statpy
A Python toolkit for statistical analysis of Markov Chain Monte Carlo data in the context of lattice QCD.  

This is the actively developed version of statpy. The original toolkit, written during my PhD, is preserved as the [v1.0 release](https://github.com/spieseba/statpy/releases/tag/v1.0) and on the [`v1-legacy`](https://github.com/spieseba/statpy/tree/v1-legacy) branch, and is no longer maintained.

### Prerequisites
- Python >= 3.12
- `uv` installed (see https://docs.astral.sh/uv/getting-started/)

---

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

---

### Verify installation
```bash
uv run python -c "import statpy; print(statpy.__file__)"
```
