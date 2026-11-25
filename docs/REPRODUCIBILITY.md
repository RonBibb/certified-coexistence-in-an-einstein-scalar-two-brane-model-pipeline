# Reproduction scope and commands

## Package identity and saved-record conversion

```sh
python3 -B src/verify_package.py
python3 -B src/replay/check_growth_rates.py
```

Both commands resolve inputs relative to their own files and require only the
standard library. Identity checks are not a mathematical validation of the stored
certificates. The growth check uses exact rational square comparisons.

## Numerical producer replay

The historical qualified environment is native Linux aarch64, CPython 3.14.4,
python-flint 0.9.0 and FLINT 3.6.0. Install the pinned dependency into a separate
environment; do not replace unavailable wheels silently or bypass platform checks.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r src/replay/requirements.txt
.venv/bin/python -c 'import flint; print(flint.__version__, flint.__FLINT_VERSION__)'
.venv/bin/python -B src/replay/replay.py --mode all --output /tmp/coexistence-core-new
.venv/bin/python -B src/replay/replay.py --mode robustness --output /tmp/coexistence-robustness-new
```

Choose output paths that do not exist. `all` runs anchor, pair, continuum, finite
fold and integer-pair stages. `robustness` runs witness, action interval and narrow
material box. Single central stages are also available. An `all` run feeds its
fresh continuum into the fold; a fold-only run uses the included saved continuum.

The preserved adapter defaults to one permitted CPU and accepts at most four via
`--cpus`. It uses affinity; it does not implement scheduler admission. Each stage
has a 900-second deadline by default and failure stops the sequence. Never use
Python `-O`: the certificate predicates use assertions. Computational runs require
the operator's resource allocation; cloning this repository launches nothing.

The eight stages passed historical clean-extraction replays on 8 November 2026.
Their [receipts](../src/replay/results/) and [scope statement](../src/replay/EXECUTED_SCOPE.txt)
retain their original dates and hashes. No interval producer was rerun for this
GitHub import. A fresh dependency installation, new architecture and full numerical
replay of this public projection have not been qualified by the packaging checks.

## Historical archives

`data/raw/` contains all imported native-result directories, including failed and
superseded attempts. Success must be read from individual records, not inferred
from a directory's presence. Historical scripts may retain absolute paths and
expired launch conditions. Use `src/replay/replay.py` for supported regeneration;
do not run archival launchers as if they were current commands.

`data/processed/claim-map.json` is an unchanged historical map: its paths are
relative to the original evidence tree, and some analytic/private report targets
are retained only in the private parent. Current public numerical paths are mapped
in `docs/IMPORT_MANIFEST.json`; the complete analytic argument is in the article.
`background_profiles.json` is floating-point illustration data, not a new interval
certificate. Raw manifests and runtime paths describe historical executions.
