# Certified coexistence and a spectral fold — reproducibility pipeline

Supporting code, frozen inputs, numerical records and current author PDFs for
**Certified coexistence and a spectral fold in an Einstein–scalar two-brane model**,
by Ronald Bibb, Liberty University. This is the companion to
[Paper A](https://doi.org/10.5281/zenodo.24515553).

## Read the paper

- [Article](manuscript/JMP_coexistence_article.pdf): 27 pages.
- [Supplement](manuscript/JMP_coexistence_supplement.pdf): 11 pages.
- [Accessibility descriptions](docs/ACCESSIBILITY.txt).

These are the 24 November 2026 author working PDFs, not an accepted or peer-reviewed
publication. The `JMP` filename prefix denotes a prospective journal target.
The companion has no Zenodo DOI or journal submission as of this import. The
unchanged PDFs contain some pre-deposit availability wording; this README supplies
the current public access route. No scientific content was revised for this import.

## Start with the lightweight checks

From the checkout, using Python 3 without optimization (`-O`):

```sh
python3 -B src/verify_package.py
python3 -B src/replay/check_growth_rates.py
```

The first checks the public file inventory, checksums and preserved replay bindings.
The second verifies the saved pair's exact growth-rate conversions. Neither reruns
an interval producer. Scientific verification for this import was performed on Linux.

For actual numerical regeneration see [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
The historical qualified platform is **native Linux aarch64**, CPython 3.14.4,
python-flint 0.9.0 and FLINT 3.6.0. Other platforms are not claimed qualified.

## Repository map

| Path | Contents |
| --- | --- |
| `src/replay/` | Package-relative numerical adapter, exact inputs, pinned dependency and historical replay results |
| `data/raw/` | Scientific source/configurations and saved runs, including adverse and superseded attempts |
| `data/processed/` | Exact table inputs, growth conversions and illustrative profile data |
| `configs/` | Configuration guide; bound configurations remain beside their sources |
| `docs/` | Reproduction scope, provenance and accessibility descriptions |
| `manuscript/` | Current generated article and supplement PDFs |
| `tests/` | Description of the available validation commands |

The adapter and numerical sources are imported byte-for-byte. Its public
`SOURCE_BINDINGS.json` is a new inventory restricted to the public replay files;
the original reviewer manifest is preserved separately. See
[the provenance note](docs/PROVENANCE.md) and [the import map](docs/IMPORT_MANIFEST.json).
The Git checkout uses ordinary Git objects; Git LFS is not required.

## Scientific limits

The paper establishes stationary coexistence and generic scalar radial spectral
counts for its declared action/domain, plus a separately local spectral fold.
It does not establish continuation across the gaps, nonlinear stability,
dynamical branch selection or event-horizon formation. Historical replay success
and internal analytic scrutiny are not external human peer review.

Editable manuscripts, historical review packages and research administration
remain in a separate private repository. The public package is usable without it.
No release tag or new scientific solver execution is implied by this GitHub import.
