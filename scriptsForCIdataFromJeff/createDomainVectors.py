#!/usr/bin/env python3
"""
Write domainVectors.inp containing:
80.0 0.0 0.0
0.0 80.0 0.0
0.0 0.0 80.0

Usage:
  python write_domain_vectors.py            # writes ./domainVectors.inp
  python write_domain_vectors.py /path/dir  # writes /path/dir/domainVectors.inp
"""

from __future__ import annotations

import argparse
from pathlib import Path


CONTENT = """80.0 0.0 0.0
0.0 80.0 0.0
0.0 0.0 80.0
"""


def main():
    ap = argparse.ArgumentParser(description="Write domainVectors.inp (80x80x80).")
    ap.add_argument("path", nargs="?", default=".", help="Output directory (default: .)")
    args = ap.parse_args()

    out_dir = Path(args.path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "domainVectors.inp").write_text(CONTENT, encoding="utf-8")


if __name__ == "__main__":
    main()
