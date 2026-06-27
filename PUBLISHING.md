# Publishing rejectkit to PyPI

Once published, anyone can `pip install rejectkit`. The package name `rejectkit`
was free on PyPI at the time of writing.

## 0. One-time setup
- Create a PyPI account: https://pypi.org/account/register/
- Create an API token (Account settings → API tokens). It looks like `pypi-AgE…`.
- Install the tooling:

```bash
pip install build twine
```

## 1. Before each release
- Fill in the real values in `pyproject.toml` (`authors`, `[project.urls]`) and
  `LICENSE` (copyright holder).
- Bump `version` in `pyproject.toml` (e.g. `0.3.0` → `0.3.1`) and add a
  `CHANGELOG.md` entry. **PyPI will not let you re-upload an existing version.**
- (Optional) Remove the hard-coded data path in
  `examples/real_data_home_credit.ipynb` before publishing.

## 2. Build the distributions

```bash
rm -rf dist build
python -m build          # creates dist/*.whl and dist/*.tar.gz
python -m twine check dist/*   # must say PASSED
```

## 3. (Recommended) Test on TestPyPI first

```bash
python -m twine upload --repository testpypi dist/*
# then, in a fresh virtualenv:
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rejectkit
```

## 4. Publish to the real PyPI

```bash
python -m twine upload dist/*
# username: __token__
# password: your pypi-… API token
```

Or non-interactively:

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-XXXX python -m twine upload dist/*
```

## 5. Verify

```bash
pip install rejectkit
python -c "import rejectkit; print(rejectkit.__version__)"
```

## Notes
- The wheel ships **only the `rejectkit` package code** (verified). Example data
  (Home Credit, etc.) is **not** included — users bring their own; the synthetic
  `rejectkit.datasets.make_credit_data` is bundled so they can try it immediately.
- Optional extras install with `pip install "rejectkit[plot]"`, `"[polars]"`, `"[docs]"`, `"[dev]"`.
- Publishing a version to PyPI is effectively permanent; you can *yank* a bad
  release but not silently replace it. Get the metadata right first.
