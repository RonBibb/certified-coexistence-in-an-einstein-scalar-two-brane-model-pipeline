#!/usr/bin/env python3
"""Read-only package and replay-binding verification; no numerical producer."""
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def safe_path(base, name):
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Manifest path escapes package: ' + name)
    return path
def verify(base, bindings):
    for name, expected in bindings.items():
        path = safe_path(base, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Missing or changed file: ' + name)

manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
verify(ROOT, manifest['sha256'])
replay = ROOT / 'src/replay'
bindings = json.loads((replay / 'SOURCE_BINDINGS.json').read_text())['sha256']
verify(replay, bindings)
original = json.loads((ROOT / 'docs/provenance/ORIGINAL_REVIEWER_SOURCE_BINDINGS.json').read_text())['sha256']
for name, value in bindings.items():
    if name in original and value != original[name]:
        raise ValueError('Original reviewer binding changed: ' + name)
imports = json.loads((ROOT / 'docs/IMPORT_MANIFEST.json').read_text())['files']
for row in imports:
    if digest(safe_path(ROOT, row['public_path'])) != row['sha256']:
        raise ValueError('Import identity mismatch: ' + row['public_path'])
print(json.dumps({'pass': True, 'package_files': len(manifest['sha256']),
                  'replay_bindings': len(bindings), 'unchanged_imports': len(imports),
                  'scientific_integrations': 0}))
