"""Resolve hash-pinned vLLM AST fixtures from this TSMC artifact only."""
import hashlib
import json
from pathlib import Path


REFERENCE = Path(__file__).resolve().parents[2] / "native_runtime_reference"


def native_source(upstream_path):
    """Return a checked fixture; never search ancestors or installed packages."""
    manifest = json.loads((REFERENCE / "MANIFEST.json").read_text())
    entry = manifest["files"][upstream_path]
    path = REFERENCE / entry["file"]
    if path.parent != REFERENCE or not path.is_file():
        raise RuntimeError(f"required bundled vLLM fixture is missing: {upstream_path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"bundled vLLM fixture hash mismatch: {upstream_path}")
    return path
