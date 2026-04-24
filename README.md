# statpy
RQCD version of statpy

### Prerequisites
- Python >= 3.12
- `uv` installed (see https://docs.astral.sh/uv/getting-started/)

---

### Regular install (non-editable)
Use this if you just want to **use** `statpy-rqcd` without modifying it.

From your project directory:

```bash
uv venv
uv pip install /absolute/path/to/statpy-rqcd
```

### Development install (editable)
Use this if you actively develop statpy-rqcd and want changes
to be picked up immediately by consuming projects.

From your project directory:

```bash
uv venv 
uv pip install meson meson-python ninja
uv add --editable /absolute/path/to/statpy-rqcd                       # Record the dependency (so version constraints are resepected)
uv pip install -e /absolute/path/to/statpy-rqcd --no-build-isolation  # Reinstall using project environment for build tools
```
The build tools are installed into the project environment and reused for editable builds.

---

### Verify installation
```bash
uv run python -c "import statpy; print(statpy.__file__)"
```
