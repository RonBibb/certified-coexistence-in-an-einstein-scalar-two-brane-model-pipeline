# Validation entry points

Run `python3 -B src/verify_package.py` for package identity and source-binding
checks, then `python3 -B src/replay/check_growth_rates.py` for exact saved growth
conversions. Both accept execution from any working directory.

The numerical replay modes are documented in `docs/REPRODUCIBILITY.md`; they are
not automatically launched by the lightweight checks. Historical failed controls
remain available in `data/raw/`.
