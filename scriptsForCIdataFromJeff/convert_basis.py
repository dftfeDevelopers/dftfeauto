"""
Convert a single-element basis block like:

H     0
S    1   1.00
  7.977000D-01  1.000000D+00
...
****
to:

#H     0
H S
  7.977000E-01  1.000000E+00
...

Rules:
- First nonblank line is the element header (e.g., "H     0"); it becomes commented with '#'.
- Each shell header line like "S    6   1.00" becomes "{Element} {Shell}" on its own line.
- The following N primitive lines are copied until the next shell header or "****".
- For numeric tokens, replace Fortran 'D' exponent with 'E' (e.g., 7.97D-01 -> 7.97E-01).
- Output file name defaults to "{element_lower}_basis.bas" unless --out is provided.
"""

import argparse
import re
from pathlib import Path

SHELL_RE = re.compile(r"^\s*([SPDFGHI])\s+(\d+)\s+([0-9.]+)\s*$", re.IGNORECASE)
END_RE   = re.compile(r"^\s*\*{4}\s*$")


def d_to_e(s: str) -> str:
    # Replace D/d with E in scientific notation tokens.
    return re.sub(r"([0-9])([Dd])([+-]?\d+)", r"\1E\3", s)


def convert_basis_block(text: str) -> tuple[str, str]:
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    # drop leading blank lines
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i == len(lines):
        raise ValueError("Input is empty/blank.")

    header = lines[i].rstrip()
    m = re.match(r"^\s*([A-Za-z]{1,2})\b", header)
    if not m:
        raise ValueError("First line must start with an element symbol, e.g. 'H     0'.")
    elem = m.group(1)

    out = []
    out.append("#" + header.strip() + "\n")

    i += 1
    while i < len(lines):
        ln = lines[i].rstrip()
        if END_RE.match(ln):
            break
        if not ln.strip():
            i += 1
            continue

        sm = SHELL_RE.match(ln)
        if sm:
            shell = sm.group(1).upper()
            nprim = int(sm.group(2))
            out.append(f"{elem} {shell}\n")

            # copy next nprim lines as primitives
            for k in range(nprim):
                i += 1
                if i >= len(lines):
                    raise ValueError(f"Unexpected EOF while reading {nprim} primitives for shell {shell}.")
                prim = lines[i].rstrip()
                if END_RE.match(prim) or SHELL_RE.match(prim):
                    raise ValueError(f"Expected primitive line, got '{prim}'.")
                out.append(d_to_e(prim).rstrip() + "\n")

            i += 1
            continue

        # If we hit a non-shell line, that's unexpected for this format.
        raise ValueError(f"Unexpected line (not a shell header): '{ln}'")

    return elem, "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="Input file containing one element basis block (ending with '****').")
    ap.add_argument("--out", help="Output filename. Default: {element_lower}_basis.bas")
    args = ap.parse_args()

    text = Path(args.infile).read_text(encoding="utf-8", errors="replace")
    elem, converted = convert_basis_block(text)

    outname = args.out if args.out else f"{elem.lower()}_basis_nwchem.bas"
    Path(outname).write_text(converted, encoding="utf-8")
    print(f"Wrote {outname}")

if __name__ == "__main__":
    main()
