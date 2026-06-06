# statpy
A Python toolkit for statistical analysis of Markov Chain Monte Carlo data in the context of lattice QCD.  

This is the actively developed version of statpy. The original toolkit, written during my PhD, is preserved as the [v1.0 release](https://github.com/spieseba/statpy/releases/tag/v1.0) and on the [`v1-legacy`](https://github.com/spieseba/statpy/tree/v1-legacy) branch, and is no longer maintained.

### Prerequisites
- Python >= 3.12
- `uv` installed (see https://docs.astral.sh/uv/getting-started/)

---

### Regular install (non-editable)
Use this if you just want to **use** `statpy` without modifying it.

From your project directory:

```bash
uv venv
uv pip install /absolute/path/to/statpy
```

### Development install (editable)
Use this if you actively develop statpy and want changes
to be picked up immediately by consuming projects.

From your project directory:

```bash
uv venv 
uv pip install meson meson-python ninja
uv add --editable /absolute/path/to/statpy                       # Record the dependency (so version constraints are respected)
uv pip install -e /absolute/path/to/statpy --no-build-isolation  # Reinstall using project environment for build tools
```
The build tools are installed into the project environment and reused for editable builds.

The final `--no-build-isolation` reinstall is required: meson-python editable
installs rebuild via `ninja` on every import, so the build must use the venv's
own `ninja` (a stable path) rather than the throwaway isolated-build one.

---

### Verify installation
```bash
uv run python -c "import statpy; print(statpy.__file__)"
```
