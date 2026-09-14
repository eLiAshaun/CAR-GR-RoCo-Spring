#!/usr/bin/env python3
"""Fetch and verify the unchanged historical CAR-GR bundle (Python >= 3.12)."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tarfile
import urllib.request

NAME = "CAR-GR-WAFT-step2000-repro-20260828.tar.gz"
ROOT_NAME = NAME.removesuffix(".tar.gz")
URL = "https://github.com/eLiAshaun/CAR-GR-RoCo-Spring/releases/download/v1.0.0/" + NAME
SHA256 = "a2d8b2dc4fe3beecd983d15f6698f355e84db056fce06da101d7c20993e159f6"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing local archive instead of downloading")
    parser.add_argument("--destination", type=Path, default=Path("reproduction_artifacts"))
    parser.add_argument("--verify-only", action="store_true", help="Check the archive SHA-256 without extracting")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or newer is required for filtered tar extraction")
    archive = args.archive
    if archive is None:
        args.destination.mkdir(parents=True, exist_ok=True)
        archive = args.destination / NAME
        if not archive.exists():
            temporary = archive.with_name(archive.name + ".part")
            print("Downloading", URL, flush=True)
            try:
                with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as stream:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        stream.write(block)
                if digest(temporary) != SHA256:
                    raise RuntimeError("Downloaded archive SHA-256 mismatch")
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
    if not archive.is_file():
        parser.error(f"Archive does not exist: {archive}")
    if digest(archive) != SHA256:
        raise RuntimeError("Archive SHA-256 mismatch; refusing to extract")
    print("Archive SHA-256: OK", flush=True)
    if args.verify_only:
        return
    output = args.destination / ROOT_NAME
    if output.exists():
        parser.error(f"Destination already exists: {output}; use another --destination")
    args.destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] != ROOT_NAME or ".." in parts or member.name.startswith("/"):
                raise RuntimeError(f"Unexpected archive member: {member.name}")
        bundle.extractall(args.destination, filter="data")
    count = 0
    for line in (output / "BUNDLE_MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split(None, 1)
        candidate = (output / relative).resolve()
        if not candidate.is_relative_to(output.resolve()):
            raise RuntimeError(f"Unsafe manifest path: {relative}")
        if not candidate.is_file() or digest(candidate) != expected:
            raise RuntimeError(f"Manifest mismatch: {relative}")
        count += 1
    print(f"Internal manifest: {count} files OK")
    print(f"Bundle ready: {output.resolve()}")
    print("Next: prepare benchmark data and follow the bundled README.")


if __name__ == "__main__":
    main()
