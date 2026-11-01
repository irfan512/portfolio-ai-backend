"""Download all-MiniLM-L6-v2 (ONNX export + tokenizer + pipeline config) at a pinned
revision and verify every file against all-MiniLM-L6-v2.sha256. Used by the Dockerfile.

    python models/fetch_model.py <revision> <checksum-file> <target-dir>
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "sentence-transformers/all-MiniLM-L6-v2"


def main(revision: str, checksums: Path, target: Path) -> None:
    expected = dict(
        reversed(line.split(maxsplit=1))
        for line in checksums.read_text().splitlines()
        if line.strip()
    )
    for name, digest in expected.items():
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPO}/resolve/{revision}/{name}"
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed https host)
        actual = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual != digest:
            sys.exit(f"Checksum mismatch for {name}: {actual} != {digest}")
        print(f"ok  {name}")
    (target / "REVISION").write_text(revision + "\n")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
