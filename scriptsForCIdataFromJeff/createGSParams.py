#!/usr/bin/env python3
"""
Create an input file with NATOMS and NATOM TYPES inferred from coordinates.inp.

- NATOMS = number of non-empty, non-comment lines in coordinates.inp
- NATOM TYPES = number of unique atomic numbers in the first column

Assumes coordinates.inp lines look like:
  atomic_number atomic_number x y z
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_coordinates_inp(path: Path) -> tuple[int, int]:
    natoms = 0
    types = set()

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("#", "!", "//")):
            continue

        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Bad line in {path}: {raw!r}")

        try:
            z1 = int(parts[0])
        except ValueError as e:
            raise ValueError(f"Could not parse atomic number from line: {raw!r}") from e

        natoms += 1
        types.add(z1)

    if natoms == 0:
        raise ValueError(f"No atoms found in {path}")

    return natoms, len(types)


def build_text(natoms: int, natom_types: int) -> str:
    # Keep formatting/indentation as in the example
    return f"""set SOLVER MODE = GS
set VERBOSITY = 5
set USE GPU = true
set WRITE STRUCTURE ENERGY FORCES DATA POST PROCESS = true
subsection Geometry
 set NATOMS = {natoms}
 set NATOM TYPES = {natom_types}
 set ATOMIC COORDINATES FILE = coordinates.inp
 set DOMAIN VECTORS FILE = domainVectors.inp
end
subsection Boundary conditions
 set PERIODIC1 = false
 set PERIODIC2 = false
 set PERIODIC3 = false
end
subsection GPU
 set USE GPU = false
 set USE ELPA GPU KERNEL = false
end
subsection Post-processing Options
 set PRINT KINETIC ENERGY  = true
 set WRITE DENSITY QUAD DATA = false
end
subsection Finite element mesh parameters
 set POLYNOMIAL ORDER = 5
 set POLYNOMIAL ORDER ELECTROSTATICS = 7
 subsection Auto mesh generation parameters
  set MESH SIZE AT ATOM = 0.03
  set MESH SIZE AROUND ATOM = 0.45
  set ATOM BALL RADIUS = 6.0
  set INNER ATOM BALL RADIUS = 0.5
 end
end
subsection DFT functional parameters
 set EXCHANGE CORRELATION TYPE = LDA-PW
 set PSEUDOPOTENTIAL CALCULATION = false
 set SPIN POLARIZATION = 0
 set START MAGNETIZATION = 0
end
subsection SCF parameters
 set MIXING PARAMETER = 0.2
 set MIXING METHOD = ANDERSON
 set COMPUTE ENERGY EACH ITER = false
 set STARTING WFC = RANDOM
 set TEMPERATURE = 10
 set TOLERANCE = 1e-05
 subsection Eigen-solver parameters
  set CHEBYSHEV FILTER TOLERANCE = 0.0001
  set ORTHOGONALIZATION TYPE = CGS
 end
end
"""


def main():
    ap = argparse.ArgumentParser(description="Generate the solver input file using coordinates.inp.")
    ap.add_argument("path", nargs="?", default=".", help="Directory containing coordinates.inp (default: .)")
    ap.add_argument("-o", "--output", default="gsParams.prm", help="Output filename (default: input.inp)")
    args = ap.parse_args()

    folder = Path(args.path).expanduser().resolve()
    coord_path = folder / "coordinates.inp"
    if not coord_path.exists():
        raise FileNotFoundError(f"Expected file not found: {coord_path}")

    natoms, natom_types = parse_coordinates_inp(coord_path)

    out_path = (folder / args.output).resolve()
    out_path.write_text(build_text(natoms, natom_types), encoding="utf-8")


if __name__ == "__main__":
    main()
