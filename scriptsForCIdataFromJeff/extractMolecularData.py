"""
Parse ONLY input #1 (everything before a line starting with '@@@@') from a Q-Chem-style file.

Outputs:
  - molecule.xyz
  - h_basis.bas
  - f_basis.bas

Basis filenames are determined by element symbol (lowercased): "{element_lower}_basis.bas".
"""

from pathlib import Path
import re
import sys

SEP_RE = re.compile(r"^\s*@@@@\s*$")
BLOCK_START_RE = re.compile(r"^\s*\$(\w+)\s*$", re.IGNORECASE)
BLOCK_END_RE = re.compile(r"^\s*\$end\s*$", re.IGNORECASE)
BASIS_ELEM_RE = re.compile(r"^\s*([A-Za-z]{1,2})\s+0\s*$")  # e.g., "H     0"
BASIS_END_ELEM_RE = re.compile(r"^\s*\*{4}\s*$")            # "****"


def first_input_text(full_text: str) -> str:
    """Return the text before the first @@@@ separator (or all text if none)."""
    out_lines = []
    for line in full_text.splitlines(True):
        if SEP_RE.match(line):
            break
        out_lines.append(line)
    return "".join(out_lines)


def extract_block(text: str, block_name: str) -> list[str]:
    """Extract lines inside $block_name ... $end (first occurrence)."""
    lines = text.splitlines()
    in_block = False
    collected = []
    for line in lines:
        if not in_block:
            m = BLOCK_START_RE.match(line)
            if m and m.group(1).lower() == block_name.lower():
                in_block = True
            continue
        else:
            if BLOCK_END_RE.match(line):
                return collected
            collected.append(line)
    raise ValueError(f"Block ${block_name} not found (or missing $end).")


def write_molecule_xyz(mol_lines: list[str], out_path: Path) -> None:
    """
    mol_lines are the lines between $molecule and $end.
    Expected:
      line0: "charge multiplicity"
      subsequent: "Element x y z"
    """
    # Strip empty/comment lines
    clean = [ln.strip() for ln in mol_lines if ln.strip() and not ln.strip().startswith("!")]
    if not clean:
        raise ValueError("Empty $molecule block.")

    # First line: charge multiplicity
    parts = clean[0].split()
    if len(parts) < 2:
        raise ValueError("First line of $molecule must be: charge multiplicity")
    charge, mult = parts[0], parts[1]

    atoms = []
    for ln in clean[1:]:
        toks = ln.split()
        if len(toks) < 4:
            continue
        el = toks[0]
        x, y, z = toks[1], toks[2], toks[3]
        atoms.append((el, x, y, z))

    if not atoms:
        raise ValueError("No atoms found in $molecule block.")

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"charge={charge} multiplicity={mult}\n")
        for el, x, y, z in atoms:
            f.write(f"{el} {x} {y} {z}\n")


def extract_basis_by_element(basis_lines: list[str]) -> dict[str, list[str]]:
    """
    Parse a $basis block (lines between $basis and $end) into per-element sections.
    Returns dict: element -> lines (including the element header line and trailing '****').
    """
    per_elem: dict[str, list[str]] = {}
    cur_elem = None
    cur_buf: list[str] = []

    for raw in basis_lines:
        line = raw.rstrip("\n")

        m = BASIS_ELEM_RE.match(line)
        if m:
            # flush previous (should have ended with ****, but be tolerant)
            if cur_elem is not None and cur_buf:
                per_elem[cur_elem] = cur_buf[:]
            cur_elem = m.group(1)
            cur_buf = [line + "\n"]
            continue

        if cur_elem is None:
            # skip anything before first "X 0"
            continue

        cur_buf.append(line + "\n")

        if BASIS_END_ELEM_RE.match(line):
            per_elem[cur_elem] = cur_buf[:]
            cur_elem = None
            cur_buf = []

    # If file ends without **** for last element, still flush
    if cur_elem is not None and cur_buf:
        per_elem[cur_elem] = cur_buf[:]

    if not per_elem:
        raise ValueError("No element basis sections found in $basis block.")

    return per_elem


def main(infile: str) -> None:
    text = Path(infile).read_text(encoding="utf-8", errors="replace")
    text1 = first_input_text(text)

    # Molecule -> molecule.xyz
    mol_lines = extract_block(text1, "molecule")
    write_molecule_xyz(mol_lines, Path("molecule.xyz"))

    # Basis -> element files
    basis_lines = extract_block(text1, "basis")
    per_elem = extract_basis_by_element(basis_lines)

    for elem, lines in per_elem.items():
        out_name = f"{elem.lower()}_basis.bas"  # element-determined filename
        Path(out_name).write_text("".join(lines), encoding="utf-8")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} input.in")
    main(sys.argv[1])
