# Public projection provenance

This public repository was assembled on 24 November 2026 from the existing
companion paper folder. The private parent retains every substantive original,
including previous deliveries, unpublished manuscript sources, internal reviews,
research administration and original reviewer ZIPs. Build caches and temporary
renderings are omitted from Git, not deleted locally.

`IMPORT_MANIFEST.json` maps each copied public file to its original relative path
and SHA-256 digest. Hashing and identity checks are performed on native Linux.
`PACKAGE_MANIFEST.json` binds the complete public package except itself.

The replay's code, numerical inputs, pinned dependency, growth check and original
results are copied unchanged. The original reviewer `SOURCE_BINDINGS.json` also
bound editable manuscripts and obsolete reading PDFs. Publishing that manifest as
the active replay binding would require publishing private manuscript sources.
Instead, `src/replay/SOURCE_BINDINGS.json` binds only the exported replay files,
with identical hashes for those entries that occurred in the original manifest.
The original manifest is retained at
`docs/provenance/ORIGINAL_REVIEWER_SOURCE_BINDINGS.json` for comparison. There is no
change to the numerical algorithms or the adapter's checks.

`ANCHOR_TO_THEOREM.txt`, historical receipts, and imported metadata are dated
provenance. Their document names, page locators and local machine paths are not
current public navigation. Current PDFs are in `manuscript/`; obsolete reviewer
reading copies have not been promoted to the current article.

The import does not rewrite old delivery manifests, claim a new numerical replay,
or publish the companion to Zenodo or a journal.
