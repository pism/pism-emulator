# Building the documentation

The documentation you are reading is Sphinx + MyST, set up like
[pism-terra's](https://github.com/pism/pism-terra). It builds from
`docs/source/`, and you can build the same thing locally before pushing.

## Environment

Install the documentation toolchain into the `pism-emulator` environment:

```bash
conda activate pism-emulator
python -m pip install -r docs/requirements.txt
```

## Build

```bash
cd docs
make html
open _build/html/index.html
```

While editing, `make livehtml` rebuilds on save and serves the result at
<http://127.0.0.1:8000>:

```bash
cd docs
make livehtml
```

`make clean` removes the build and the generated API pages.

## Read the Docs

The API pages import the package, so `.readthedocs.yaml` installs its
dependencies for real: CPU-only PyTorch first (the default Linux wheel pulls in
CUDA), then `requirements.txt`, `docs/requirements.txt` and the package itself.
Mocking PyTorch instead does not work: SciPy then mistakes the mock for a real
`torch` and fails when `pism_emulator.lhs.draw` imports `scipy.stats`.

## Writing pages

- Pages are MyST Markdown (`.md`). Sphinx directives use fenced blocks such as
  ` ```{list-table} ` or ` ```{admonition} `, and cross-references use
  `` {doc}`../features/emulators` ``, `` {class}`~pism_emulator.mcmc.mala.MALASamplerModule` ``
  or `` {cite:t}`Aschwanden2022` ``.
- Citations live in `docs/source/refs.bib`.
- New public functions and classes go into the autosummary tables in
  `reference/api.md`; their numpydoc docstrings become the API pages.
- Mark missing content with a `TODO` admonition, which the theme shows in
  orange.
