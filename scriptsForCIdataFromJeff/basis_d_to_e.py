#!/usr/bin/env python3
"""
Convert a Gaussian-style basis file from Fortran 'D' exponents to 'E',
and remove trailing "****" separators.

Reads:  {element_id}_basis.bas
Writes: {element_id}_basis_invdft.bas

Usage:
  python basis_d_to_e.py H1
  python basis_d_to_e.py /path/to/H1_basis.bas
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

# Replace D/d used as exponent marker with E, only when it looks like scientific notation:
# e.g., 7.977000D-01 -> 7.977000E-01
SCI_D_RE = re.compile(r"(?P<mant>\d(?:[\d\.]*))(?P<d>[Dd])(?P<exp>[+-]\d+)")

def convert_line(line: str) -> str:
    return SCI_D_RE.sub(r"\g<mant>E\g<exp>", line)

def strip_trailing_stars(lines: list[str]) -> list[str]:
    """
    Remove trailing lines that are exactly '****' (optionally with whitespace).
    Keeps interior '****' if present; only strips from the end of the file.
    """
    i = len(lines)
    while i > 0 and lines[i - 1].strip() == "****":
        i -= 1
    return lines[:i]

def resolve_input(arg: str) -> tuple[Path, Path]:
    p = Path(arg).expanduser()
    if p.suffix:  # user passed a filename like H1_basis.bas
        in_path = p
        if in_path.name.endswith("_basis.bas"):
            out_name = in_path.name.replace("_basis.bas", "_basis_invdft.bas")
        else:
            out_name = in_path.stem + "_invdft" + in_path.suffix
        out_path = in_path.with_name(out_name)
        return in_path, out_path

    # user passed an element_id like H1
    in_path = Path(f"{arg}_basis.bas")
    out_path = Path(f"{arg}_basis_invdft.bas")
    return in_path, out_path

def main():
    ap = argparse.ArgumentParser(description="Convert D-exponent basis file to E-exponent and strip trailing ****.")
    ap.add_argument("element_id_or_file", help="Element id (e.g., H1) or path to *_basis.bas")
    args = ap.parse_args()

    in_path, out_path = resolve_input(args.element_id_or_file)
    in_path = in_path.expanduser().resolve()
    out_path = out_path.expanduser().resolve()

    raw_lines = in_path.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    raw_lines = strip_trailing_stars(raw_lines)

    converted = "".join(convert_line(ln) for ln in raw_lines)
    out_path.write_text(converted, encoding="utf-8")

if __name__ == "__main__":
    main()
